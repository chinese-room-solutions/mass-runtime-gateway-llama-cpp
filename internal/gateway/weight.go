package gateway

import (
	"encoding/binary"
	"math"

	llamacpp "github.com/chinese-room-solutions/mass-proto/gen/go/llama-cpp"
)

// bytesPerToken is the well-known ~4 bytes/token rule for English; it
// under-counts code and non-ASCII but that's fine for the
// input-token-count signal MASS uses for scoring.
const bytesPerToken = 4

// Vision-projector token estimate. An image's prefill cost is driven by
// how many patch tokens the vision encoder emits, which is a function of
// the image's pixel dimensions — NOT its file byte size. Counting raw
// image bytes as text tokens over-estimated by ~100× (a 1.4 MB page →
// ~350k phantom tokens → a ~30-minute prediction for a job that runs in
// seconds). The true count needs the mmproj loaded (mtmd_tokenize on a
// worker), which isn't available at Submit time, so we approximate from
// dimensions read straight from the image header.
//
// tokens ≈ (effW/patch) × (effH/patch) / merge
//
// where effW×effH is the image clamped to the encoder's pixel budget —
// NOT the raw header dimensions. Vision encoders smart-resize an image
// down before tiling, so a high-DPI scan tiles to a bounded token
// count regardless of source resolution. Using raw dimensions
// over-counted a 300-DPI page by ~9×, which dominated the prefill
// estimate and pinned the scheduler's correction EWMA at its clamp
// ceiling.
//
// The budget is expressed in output TOKENS, mirroring llama.cpp:
// tools/mtmd/clip.cpp caps the Qwen-VL family (and most current
// encoders) with set_limit_image_tokens(8, 4096), i.e. pixels saturate
// at visionMaxTokens × patch² × merge. For a patch-16/merge-4 model
// that is ~4.2M px — an earlier fixed ~1.0M px budget under-counted
// full-page scans (the pdf2html workload) by ~3×.
//
// patch and merge default to the Qwen2/2.5-VL shape (14-pixel patches,
// 4× pixel-unshuffle merge). When the submit carries a companion mmproj
// the values read from its GGUF header override the defaults — see
// [visionParams]; Qwen3-VL moved to 16-pixel patches, a (16/14)² ≈ 1.3×
// difference the metadata resolves exactly. The scheduler's
// per-(worker,axis) throughput EWMA corrects any residual bias from
// real completions.
const (
	visionPatchPixels  = 14
	visionMergeFactor  = 4
	visionMaxTokens    = 4096 // llama.cpp clip.cpp set_limit_image_tokens upper bound
	imageTokensFloor   = 16   // a tiny/unparseable image still costs a few patches.
	imageTokensDefault = 768  // dimensions unreadable → a typical single-tile image.
	audioTokensDefault = 1500 // ~30 s of audio at typical encoder rates; coarse.
)

// visionParams is the vision-encoder shape for the dispatch's model,
// read from the companion mmproj's GGUF metadata via the catalogue
// (see [handlers.visionParams]). The zero value falls back to the
// package defaults above.
type visionParams struct {
	patchPixels int // clip.vision.patch_size
	mergeFactor int // clip.vision.spatial_merge_size², patches per output token
}

func (vp visionParams) patch() int {
	if vp.patchPixels > 0 {
		return vp.patchPixels
	}
	return visionPatchPixels
}

func (vp visionParams) merge() int {
	if vp.mergeFactor > 0 {
		return vp.mergeFactor
	}
	return visionMergeFactor
}

// maxPixels is the smart-resize budget: the pixel area at which the
// encoder's output saturates at visionMaxTokens.
func (vp visionParams) maxPixels() int {
	return visionMaxTokens * vp.patch() * vp.patch() * vp.merge()
}

// q4kMatvecAxis is the throughput-axis name every llama-cpp worker is
// required to bench. Declared in InitResponse.DefaultCostAxis so MASS
// can fall back to it when a Submit names an axis the worker hasn't
// measured.
const q4kMatvecAxis = "q4k_matvec"

// flopsPerTokenPerParam is the well-known transformer forward-pass
// physics constant: roughly 2 FLOPs per token per parameter, one
// multiply plus one add. The cost emitted by [predictCost] has units
// of FLOPs / 1e9 (i.e. GFLOPs) at decode efficiency, so MASS dividing
// by the worker's q4k_matvec GFLOPS yields predicted wall-clock
// seconds on that worker.
const flopsPerTokenPerParam = 2.0

// prefillSpeedup discounts input (prefill) tokens relative to decode
// tokens in the cost formula. The q4k_matvec bench the cost divides by
// is a batch-1 matvec chain — the memory-bound decode regime. Prefill
// processes the whole prompt as batched matmuls that run far faster
// per token on the same device: ~4-8× on CPU, ~15-50× on discrete
// GPUs. Pricing prefill at matvec throughput over-predicted
// long-prompt jobs by up to an order of magnitude — a shape-dependent
// bias the scheduler's per-(worker, axis) EWMA cannot learn away,
// since it corrects all jobs on a worker by one scalar (clamped to
// ±4×). 12 is the geometric middle of the observed range; the EWMA
// absorbs the ≤4× per-worker residual at either extreme.
const prefillSpeedup = 12

// fallbackParameterCount is used when the GGUF header didn't carry a
// readable parameter_count and the filename's size_label was missing
// or unparseable. 7B is a reasonable median across today's models —
// the operator sees an over- or underestimate proportional to the
// model's real size vs 7B.
const fallbackParameterCount uint64 = 7_000_000_000

// defaultMaxTokensCap mirrors mass-worker-llama-cpp's chat_model.cpp:
// when the request omits max_tokens, the worker uses
// min(1024, n_ctx - produced - 1), so 1024 is the headroom the worker
// will allow in practice. Anchoring the gateway's estimate to the
// worker's own default keeps the two sides honest about each other's
// behaviour.
const defaultMaxTokensCap = 1024

// predictCost returns (cost, axis) for a llama-cpp job. cost has units
// of GFLOPs at decode (matvec) efficiency; MASS divides by the worker's
// q4k_matvec throughput in GFLOPS to get predicted wall-clock seconds.
//
// Formula:
//
//	cost = (input_tokens/prefillSpeedup + expected_decode_tokens) *
//	         flopsPerTokenPerParam * parameterCount / 1e9
//
// parameterCount is the compute-relevant (per-token active) count from
// the GGUF header — see [gguf.Meta.ActiveParameterCount]; 0 falls back
// to a 7B median so unknown-size models still get a sensible estimate.
//
// expected_decode_tokens branches on the model's thinking capability:
// thinking models (Qwen3-thinking, DeepSeek-R1, gpt-oss reasoning, …)
// routinely reach the max_tokens cap on internal reasoning, so the cap
// stays as a pessimistic-but-truthful estimate. Plain chat models
// almost never hit the cap (EOS comes earlier) and reply length tracks
// prompt length within ~chatDecodeRatio×, so we use the ratio instead
// — the cap as a flat predictor over-estimated typical chat by 5-10×.
//
// Embed/batch-embed jobs are pure prefill (no decode term); tokenize
// never runs the transformer at all — the worker answers it with a
// vocab lookup — so it gets the nominal one-token floor, which also
// keeps Cost > 0 (a MASS submit invariant).
//
// Always returns axis = [q4kMatvecAxis]; future per-job axis selection
// is gated on workers learning to bench them.
func predictCost(job *llamacpp.Job, parameterCount uint64, thinking bool, vp visionParams) (float64, string) {
	if parameterCount == 0 {
		parameterCount = fallbackParameterCount
	}
	work := 1.0 // one-token floor: empty/unknown jobs still cost something
	if job.GetKind() != llamacpp.JobKind_JOB_KIND_TOKENIZE {
		input, textInput := jobInputTokens(job, vp)
		// Decode length tracks the textual prompt plus a discounted media
		// contribution (see mediaDecodeDivisor): media drives prefill in full
		// but has less leverage than text on how long the reply runs.
		decodeBase := textInput + (input-textInput)/mediaDecodeDivisor
		decode := jobExpectedDecodeTokens(job, decodeBase, thinking)
		if w := float64(input)/prefillSpeedup + float64(decode); w > work {
			work = w
		}
	}
	cost := work * flopsPerTokenPerParam * float64(parameterCount) / 1e9
	return cost, q4kMatvecAxis
}

// jobInputTokens estimates input-token count from a job's payload,
// returning the full prefill total and the text-only portion separately.
// The text portion drives the decode-length estimate (reply length tracks
// the textual prompt, not image patch count); the full total drives
// prefill cost. They differ only for multimodal chat — for all other
// kinds text == total. Returns (0, 0) on nil/unknown kinds (tokenize is
// priced without a forward pass in predictCost and never reaches here);
// predictCost floors the combined work to ≥1 token.
func jobInputTokens(job *llamacpp.Job, vp visionParams) (total, text int64) {
	if job == nil {
		return 0, 0
	}
	switch job.GetKind() {
	case llamacpp.JobKind_JOB_KIND_CHAT:
		textTokens, mediaTokens := chatInputTokens(job.GetChat(), vp)
		return textTokens + mediaTokens, textTokens
	case llamacpp.JobKind_JOB_KIND_EMBED:
		t := int64(len(job.GetEmbed().GetInput()) / bytesPerToken)
		return t, t
	case llamacpp.JobKind_JOB_KIND_BATCH_EMBED:
		sum := 0
		for _, in := range job.GetBatchEmbed().GetInputs() {
			sum += len(in)
		}
		t := int64(sum / bytesPerToken)
		return t, t
	}
	return 0, 0
}

// chatDecodeRatio estimates output-tokens as a multiple of the input
// decode-base for plain (non-thinking) chat workloads. Empirical median
// for conversational LLM usage: reply length tracks prompt length within
// ~2-3×. The full max_tokens cap is rarely hit (EOS comes earlier),
// so using the cap as a flat predictor wildly over-estimated typical
// chat. Thinking models bypass this heuristic (see jobExpectedDecodeTokens).
const chatDecodeRatio = 3

// mediaDecodeDivisor discounts media input in the decode-length estimate.
// Image/audio tokens influence reply length (a vision task emits a response
// roughly proportional to image content) but with less leverage than text:
// a page image tiles to ~1.2k tokens, yet the reply doesn't run as long as
// 1.2k tokens of prose would imply. Halving the media contribution keeps it
// scaling smoothly with image size without the ~2× over-predict that pinned
// the scheduler's correction EWMA on vision jobs (full media count) — and
// the EWMA absorbs the modest residual.
const mediaDecodeDivisor = 2

// jobExpectedDecodeTokens estimates how many output tokens a job will
// generate. Only chat jobs decode; embed/batch-embed/tokenize are
// single-shot forward passes with no autoregressive output.
//
// Two regimes:
//
//   - thinking model: generates long internal reasoning before answering,
//     regularly reaches the cap. We use the operator-set max_tokens
//     (or defaultMaxTokensCap) as the prediction — pessimistic but
//     truthful for this class.
//
//   - plain chat: reply length scales with the prompt. We use
//     min(cap, chatDecodeRatio × decodeBase) so the prediction is
//     bounded by the cap (a real ceiling) AND by the typical-reply
//     ratio (so chat predictions land near observed wall-clock).
//
// decodeBase is text tokens plus a discounted media contribution (see
// mediaDecodeDivisor in predictCost), NOT the full prefill total:
// image/audio tokens inflate prefill but only a bounded slice of them
// lengthens the reply, so feeding the combined count here over-predicted
// decode by ~2× on vision jobs (a page image tiles to ~1.2k tokens →
// 3× = thousands of phantom decode tokens the model never emits).
//
// Empty-prompt chat falls back to the cap regardless of mode — without
// any base tokens the ratio collapses to 0, which would predict near-zero
// cost and let zero-prompt jobs jump the queue ahead of real work.
func jobExpectedDecodeTokens(job *llamacpp.Job, decodeBase int64, thinking bool) int64 {
	if job == nil || job.GetKind() != llamacpp.JobKind_JOB_KIND_CHAT {
		return 0
	}
	c := job.GetChat()
	if c == nil {
		return 0
	}
	cap := int64(defaultMaxTokensCap)
	if s := c.GetSampling(); s != nil && s.GetMaxTokens() > 0 {
		cap = int64(s.GetMaxTokens())
	}
	if thinking {
		return cap
	}
	heuristic := chatDecodeRatio * decodeBase
	if heuristic <= 0 {
		return cap
	}
	if heuristic < cap {
		return heuristic
	}
	return cap
}

// chatInputTokens returns the text-token and media-token counts for a chat
// job separately. They serve different estimates: media tokens count toward
// prefill cost but not toward the decode-length ratio, since a model's reply
// length tracks the textual prompt, not how many patches an image tiles into.
func chatInputTokens(c *llamacpp.ChatJob, vp visionParams) (textTokens, mediaTokens int64) {
	if c == nil {
		return 0, 0
	}
	var textBytes int
	for _, m := range c.GetMessages() {
		textBytes += len(m.GetContent())
		for _, p := range m.GetParts() {
			textBytes += len(p.GetText())
			// Media parts are token-counted by their encoder's output, not
			// their byte size: an image's cost scales with pixel dimensions
			// (patch count), an audio clip's with duration. Byte size is a
			// near-useless proxy (PNG vs JPEG, compression) and over-counts
			// by orders of magnitude — see the vision-projector constants.
			if img := p.GetImage().GetData(); len(img) > 0 {
				mediaTokens += imageTokensFromBytes(img, vp)
			}
			if aud := p.GetAudio().GetData(); len(aud) > 0 {
				mediaTokens += audioTokensDefault
			}
		}
	}
	return int64(textBytes / bytesPerToken), mediaTokens
}

// imageTokensFromBytes estimates the vision-encoder token count for an
// encoded image from its pixel dimensions (read from the header — no
// decode), falling back to a single-tile default when the format isn't
// recognised. See the vision-projector constants for the patch math.
func imageTokensFromBytes(data []byte, vp visionParams) int64 {
	w, h, ok := imageDimensions(data)
	if !ok || w == 0 || h == 0 {
		return imageTokensDefault
	}
	w, h = clampToMaxPixels(w, h, vp.maxPixels())
	patch := vp.patch()
	patches := (w / patch) * (h / patch)
	tokens := patches / vp.merge()
	if tokens < imageTokensFloor {
		return imageTokensFloor
	}
	return int64(tokens)
}

// clampToMaxPixels scales (w, h) down — aspect preserved — so the pixel
// count does not exceed maxPixels, mirroring the smart-resize a vision
// encoder applies before tiling. Images already within budget pass
// through unchanged. Both dimensions stay ≥ 1.
func clampToMaxPixels(w, h, maxPixels int) (int, int) {
	area := w * h
	if area <= maxPixels || area <= 0 {
		return w, h
	}
	scale := math.Sqrt(float64(maxPixels) / float64(area))
	w = int(float64(w) * scale)
	h = int(float64(h) * scale)
	if w < 1 {
		w = 1
	}
	if h < 1 {
		h = 1
	}
	return w, h
}

// imageDimensions reads pixel width/height from a PNG or JPEG header
// without decoding the pixels. Returns ok=false for any other format or a
// truncated header; callers fall back to a default token estimate.
func imageDimensions(data []byte) (w, h int, ok bool) {
	switch {
	case len(data) >= 24 && binary.BigEndian.Uint64(data[0:8]) == 0x89504e470d0a1a0a:
		// PNG: the IHDR chunk follows the 8-byte signature; width/height are
		// the first two big-endian uint32s of its data (bytes 16..24).
		return int(binary.BigEndian.Uint32(data[16:20])), int(binary.BigEndian.Uint32(data[20:24])), true
	case len(data) >= 2 && data[0] == 0xFF && data[1] == 0xD8:
		return jpegDimensions(data)
	}
	return 0, 0, false
}

// jpegDimensions walks JPEG marker segments to the first Start-Of-Frame
// (SOF0–SOF15, excluding the non-frame DHT/DAC/RST markers), whose payload
// carries height then width as big-endian uint16s. Returns ok=false if no
// SOF is found before the data ends.
func jpegDimensions(data []byte) (w, h int, ok bool) {
	// Skip the leading 0xFFD8 (SOI). Segments are 0xFF, marker, len_hi, len_lo.
	for i := 2; i+9 < len(data); {
		if data[i] != 0xFF {
			i++
			continue
		}
		marker := data[i+1]
		// Standalone markers (no length): padding 0xFF, RSTn (0xD0–0xD7), SOI/EOI.
		if marker == 0xFF || (marker >= 0xD0 && marker <= 0xD9) {
			i += 2
			continue
		}
		segLen := int(binary.BigEndian.Uint16(data[i+2 : i+4]))
		if segLen < 2 {
			return 0, 0, false
		}
		// SOF markers carry frame dimensions; 0xC4 (DHT), 0xC8 (JPG), 0xCC
		// (DAC) are not frame headers despite being in the 0xC0–0xCF range.
		if marker >= 0xC0 && marker <= 0xCF && marker != 0xC4 && marker != 0xC8 && marker != 0xCC {
			if i+9 >= len(data) {
				return 0, 0, false
			}
			h = int(binary.BigEndian.Uint16(data[i+5 : i+7]))
			w = int(binary.BigEndian.Uint16(data[i+7 : i+9]))
			return w, h, true
		}
		i += 2 + segLen
	}
	return 0, 0, false
}

// kvCacheDtypeBytes is the per-element size of the KV cache. llama.cpp
// defaults to F16 (2 bytes); cache_type overrides (q8_0, q4_0) would
// shrink this. We use F16 as the upper bound so MASS's eligibility
// check biases toward "won't fit" rather than dispatching a job that
// then OOMs at load.
const kvCacheDtypeBytes = 2

// defaultContextSize mirrors DefaultContextSize in the gateway —
// 4096 when LoadHints.context_size is 0.
const defaultContextSize = 4096

// activationScratchBytes covers per-decode activations, KV-buffer
// alignment slop, and CUDA/ROCm/Vulkan working memory. 512 MiB is a
// conservative one-size-fits-all bound — small models lose some
// precision; large models stay well within. Adjust if benchmarks
// show systematic under-fit on any worker tier.
const activationScratchBytes = 512 * 1024 * 1024

// estimateLoadBytes returns the gateway's best guess at device memory
// the load will consume on the worker, split into a fixed cost paid
// once and an incremental cost paid per concurrent slot.
//
//	base    = file_bytes + activation_scratch_bytes
//	perSlot = kv_cache_bytes(props, ctx)
//
// MASS combines them with the chosen worker's free memory to project
// the actual post-grow pool size and the resulting load wall-clock.
//
// Returns (0, 0) when fileBytes is non-positive (no weights to size
// against — MASS treats 0 as "skip the eligibility check and fall
// back to pay-on-failure"). Returns (base, 0) when GGUF metadata is
// too sparse to size KV honestly; the projection collapses to pool=1.
func estimateLoadBytes(fileBytes int64, props map[string]string, contextSize int32) (base, perSlot int64) {
	if fileBytes <= 0 {
		return 0, 0
	}
	if contextSize <= 0 {
		contextSize = defaultContextSize
	}
	base = fileBytes + int64(activationScratchBytes)
	perSlot = kvCacheBytes(props, contextSize)
	return base, perSlot
}

// kvCacheBytes returns the projected KV cache footprint for one
// context slot at the given context size. Returns 0 when any required
// GGUF field is missing — callers treat that as "no per-slot term"
// and the pool-size projection downstream collapses to a single slot.
func kvCacheBytes(props map[string]string, contextSize int32) int64 {
	layers := atoiUint64(props["layers"])
	embedding := atoiUint64(props["embedding"])
	heads := atoiUint64(props["head_count"])
	if layers == 0 || embedding == 0 || heads == 0 {
		return 0
	}
	headDim := embedding / heads
	if headDim == 0 {
		return 0
	}
	// head_count_kv defaults to head_count when GGUF doesn't carry it
	// (non-GQA architectures store keys for every head).
	kvHeads := atoiUint64(props["head_count_kv"])
	if kvHeads == 0 {
		kvHeads = heads
	}
	const kAndV = 2
	return int64(kAndV) * int64(layers) * int64(contextSize) * int64(kvHeads) * int64(headDim) * int64(kvCacheDtypeBytes)
}

// atoiUint64 parses a base-10 unsigned integer string; returns 0 on
// empty or unparseable input. Used to lift props (map[string]string,
// the catalogue's runtime-agnostic shape) into the numeric form the
// memory estimator needs.
func atoiUint64(s string) uint64 {
	if s == "" {
		return 0
	}
	var n uint64
	for _, c := range s {
		if c < '0' || c > '9' {
			return 0
		}
		n = n*10 + uint64(c-'0')
	}
	return n
}
