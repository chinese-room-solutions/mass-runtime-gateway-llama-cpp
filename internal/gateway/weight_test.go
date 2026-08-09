package gateway

import (
	"bytes"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"strings"
	"testing"

	llamacpp "github.com/chinese-room-solutions/mass-proto/gen/go/llama-cpp"
	"github.com/stretchr/testify/require"
)

// predictCost spans the four llama-cpp job kinds plus edge cases. Cost
// is (input_tokens/prefillSpeedup + expected_decode_tokens) × 2 ×
// parameter_count / 1e9 (units: GFLOPs at decode efficiency — prefill
// tokens are discounted because batched prompt processing runs far
// faster than the matvec decode regime the cost divides by). Chat
// decode is bounded by min(max_tokens_cap, chatDecodeRatio × input)
// for plain chat models, and by max_tokens_cap alone for thinking
// models. Embed jobs are pure prefill; tokenize never runs the
// transformer and gets the one-token floor.
func TestPredictCost(t *testing.T) {
	const params uint64 = 1_000_000_000 // 1B for round-number arithmetic
	tokenCost := flopsPerTokenPerParam * float64(params) / 1e9
	// chatDecodeNoCap is the decode term when sampling.max_tokens is
	// unset — gateway falls back to defaultMaxTokensCap.
	chatDecodeNoCap := float64(defaultMaxTokensCap) * tokenCost
	maxTokensPtr := func(n int32) *int32 { return &n }
	tests := []struct {
		name     string
		job      *llamacpp.Job
		thinking bool
		wantCost float64
	}{
		{
			name:     "nil job uses one-token floor",
			job:      nil,
			wantCost: tokenCost,
		},
		{
			// Plain chat, 100-token prompt, default 1024 cap.
			// Heuristic: min(1024, 3×100) = 300 → decode = 300.
			name: "plain chat: decode bounded by ratio × input",
			job: &llamacpp.Job{
				Kind: llamacpp.JobKind_JOB_KIND_CHAT,
				Body: &llamacpp.Job_Chat{Chat: &llamacpp.ChatJob{
					Messages: []*llamacpp.ChatMessage{
						{Content: strings.Repeat("a", 400)},
					},
				}},
			},
			wantCost: (100.0/prefillSpeedup + 300) * tokenCost,
		},
		{
			// Plain chat, 4-token prompt, 4096-token cap.
			// Heuristic: min(4096, 3×4) = 12 → decode = 12.
			// Old behaviour predicted 4096 decode tokens; the ratio
			// heuristic predicts 12.
			name: "plain chat: short prompt + huge cap → ratio wins",
			job: &llamacpp.Job{
				Kind: llamacpp.JobKind_JOB_KIND_CHAT,
				Body: &llamacpp.Job_Chat{Chat: &llamacpp.ChatJob{
					Messages: []*llamacpp.ChatMessage{
						{Content: strings.Repeat("a", 16)},
					},
					Sampling: &llamacpp.SamplingParams{MaxTokens: maxTokensPtr(4096)},
				}},
			},
			wantCost: (4.0/prefillSpeedup + 12) * tokenCost,
		},
		{
			// Vision chat: an unrecognised-format image part (no parseable
			// dimensions) contributes imageTokensDefault input tokens, NOT
			// len(bytes)/4 — byte size is not the cost driver. With 768
			// image tokens the decode heuristic min(1024, 3×768)=1024 hits
			// the cap. Guards against the old bytes-as-tokens over-estimate.
			name: "vision chat: image sized by projector tokens, not bytes",
			job: &llamacpp.Job{
				Kind: llamacpp.JobKind_JOB_KIND_CHAT,
				Body: &llamacpp.Job_Chat{Chat: &llamacpp.ChatJob{
					Messages: []*llamacpp.ChatMessage{
						{Parts: []*llamacpp.ContentPart{
							{Content: &llamacpp.ContentPart_Image{Image: &llamacpp.ImageContent{Data: make([]byte, 4000)}}},
						}},
					},
				}},
			},
			wantCost: float64(imageTokensDefault)/prefillSpeedup*tokenCost + chatDecodeNoCap,
		},
		{
			// Vision chat, small parseable image, no text. A 280×280 PNG
			// tiles to (280/14)²/4 = 100 media tokens. Media counts in full
			// toward prefill (input=100) but is halved for the decode base
			// (mediaDecodeDivisor): decodeBase = 0 + 100/2 = 50, so
			// decode = min(1024, 3×50) = 150 — NOT 3×100 = 300. Pins the
			// media-discount path that keeps vision decode from over-counting.
			name: "vision chat: media discounted in decode base",
			job: &llamacpp.Job{
				Kind: llamacpp.JobKind_JOB_KIND_CHAT,
				Body: &llamacpp.Job_Chat{Chat: &llamacpp.ChatJob{
					Messages: []*llamacpp.ChatMessage{
						{Parts: []*llamacpp.ContentPart{
							{Content: &llamacpp.ContentPart_Image{Image: &llamacpp.ImageContent{Data: encodePNG(t, 280, 280)}}},
						}},
					},
				}},
			},
			wantCost: (100.0/prefillSpeedup + 150) * tokenCost,
		},
		{
			// Thinking model: same short prompt + huge cap as the
			// second case, but thinking=true bypasses the ratio and
			// uses the full cap. Without this branch, the heuristic
			// would under-predict reasoning workloads 100×+.
			name: "thinking chat: cap drives decode even with short prompt",
			job: &llamacpp.Job{
				Kind: llamacpp.JobKind_JOB_KIND_CHAT,
				Body: &llamacpp.Job_Chat{Chat: &llamacpp.ChatJob{
					Messages: []*llamacpp.ChatMessage{
						{Content: strings.Repeat("a", 16)},
					},
					Sampling: &llamacpp.SamplingParams{MaxTokens: maxTokensPtr(4096)},
				}},
			},
			thinking: true,
			wantCost: (4.0/prefillSpeedup + 4096) * tokenCost,
		},
		{
			// Thinking model with no max_tokens: defaultMaxTokensCap.
			name: "thinking chat: default cap when max_tokens unset",
			job: &llamacpp.Job{
				Kind: llamacpp.JobKind_JOB_KIND_CHAT,
				Body: &llamacpp.Job_Chat{Chat: &llamacpp.ChatJob{
					Messages: []*llamacpp.ChatMessage{
						{Content: strings.Repeat("a", 400)},
					},
				}},
			},
			thinking: true,
			wantCost: 100.0/prefillSpeedup*tokenCost + chatDecodeNoCap,
		},
		{
			// Empty-prompt chat (no tokens) falls back to the cap
			// regardless of mode — otherwise the heuristic would
			// predict near-zero cost and let zero-prompt jobs jump
			// the queue ahead of real work. Total tokens = 0 input
			// + cap decode; no floor needed since decode is non-zero.
			name: "plain chat: empty prompt falls back to cap",
			job: &llamacpp.Job{
				Kind: llamacpp.JobKind_JOB_KIND_CHAT,
				Body: &llamacpp.Job_Chat{Chat: &llamacpp.ChatJob{}},
			},
			wantCost: chatDecodeNoCap,
		},
		{
			name: "embed: pure prefill (input bytes/4, discounted)",
			job: &llamacpp.Job{
				Kind: llamacpp.JobKind_JOB_KIND_EMBED,
				Body: &llamacpp.Job_Embed{Embed: &llamacpp.EmbedJob{Input: strings.Repeat("x", 256)}},
			},
			wantCost: 64.0 / prefillSpeedup * tokenCost,
		},
		{
			name: "batch_embed: summed pure prefill",
			job: &llamacpp.Job{
				Kind: llamacpp.JobKind_JOB_KIND_BATCH_EMBED,
				Body: &llamacpp.Job_BatchEmbed{BatchEmbed: &llamacpp.BatchEmbedJob{
					Inputs: []string{strings.Repeat("y", 128), strings.Repeat("z", 256)},
				}},
			},
			// (128+256)/4 = 96 input tokens.
			wantCost: 96.0 / prefillSpeedup * tokenCost,
		},
		{
			// Tokenize is a vocab lookup on the worker — no forward
			// pass — so text length must not drive its cost. It gets
			// the nominal one-token floor regardless of input size.
			name: "tokenize: one-token floor, no forward pass",
			job: &llamacpp.Job{
				Kind: llamacpp.JobKind_JOB_KIND_TOKENIZE,
				Body: &llamacpp.Job_Tokenize{Tokenize: &llamacpp.TokenizeJob{Text: strings.Repeat("q", 80)}},
			},
			wantCost: tokenCost,
		},
		{
			// Embed with thinking=true must still skip decode entirely
			// — thinking only matters for chat. Defensive: we don't
			// want a wrongly-flagged embedding model to get a 1024-
			// token decode estimate slapped on.
			name: "embed: thinking flag is ignored",
			job: &llamacpp.Job{
				Kind: llamacpp.JobKind_JOB_KIND_EMBED,
				Body: &llamacpp.Job_Embed{Embed: &llamacpp.EmbedJob{Input: strings.Repeat("x", 256)}},
			},
			thinking: true,
			wantCost: 64.0 / prefillSpeedup * tokenCost,
		},
		{
			name: "tiny job floors to one token (no decode for embed)",
			job: &llamacpp.Job{
				Kind: llamacpp.JobKind_JOB_KIND_EMBED,
				Body: &llamacpp.Job_Embed{Embed: &llamacpp.EmbedJob{Input: "hi"}},
			},
			// 2/4 = 0 input tokens → floored to 1 token of work.
			wantCost: tokenCost,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotCost := predictCost(tt.job, params, tt.thinking, visionParams{})
			require.InDelta(t, tt.wantCost, gotCost, 1e-9)
		})
	}
}

// encodePNG / encodeJPEG produce real encoded headers so imageDimensions
// is exercised against the actual byte layout, not hand-rolled bytes.
func encodePNG(t *testing.T, w, h int) []byte {
	t.Helper()
	var buf bytes.Buffer
	require.NoError(t, png.Encode(&buf, image.NewRGBA(image.Rect(0, 0, w, h))))
	return buf.Bytes()
}

func encodeJPEG(t *testing.T, w, h int) []byte {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	img.Set(0, 0, color.White)
	var buf bytes.Buffer
	require.NoError(t, jpeg.Encode(&buf, img, nil))
	return buf.Bytes()
}

// imageDimensions reads width/height from PNG and JPEG headers and rejects
// anything else, so the projector-token estimate tracks pixel size.
func TestImageDimensions(t *testing.T) {
	t.Run("png", func(t *testing.T) {
		w, h, ok := imageDimensions(encodePNG(t, 640, 480))
		require.True(t, ok)
		require.Equal(t, 640, w)
		require.Equal(t, 480, h)
	})
	t.Run("jpeg", func(t *testing.T) {
		w, h, ok := imageDimensions(encodeJPEG(t, 800, 600))
		require.True(t, ok)
		require.Equal(t, 800, w)
		require.Equal(t, 600, h)
	})
	t.Run("unrecognised format", func(t *testing.T) {
		_, _, ok := imageDimensions([]byte("not an image"))
		require.False(t, ok)
	})
	t.Run("truncated header", func(t *testing.T) {
		_, _, ok := imageDimensions([]byte{0x89, 0x50, 0x4e})
		require.False(t, ok)
	})
}

// imageTokensFromBytes scales with pixel dimensions (patch count), floors
// tiny images, and falls back to a single-tile default when the format is
// unreadable — never the old bytes/4 over-estimate.
func TestImageTokensFromBytes(t *testing.T) {
	// 1120×1120 = 1.25M px; whether it clamps depends on the default
	// budget — compute the expectation through the same clamp either way.
	// The estimate is the clamped patch count, and it stays independent of
	// the PNG's byte size (which is what the old code wrongly used).
	big := encodePNG(t, 1120, 1120)
	cw, ch := clampToMaxPixels(1120, 1120, visionParams{}.maxPixels())
	wantBig := int64((cw / visionPatchPixels) * (ch / visionPatchPixels) / visionMergeFactor)
	require.Equal(t, wantBig, imageTokensFromBytes(big, visionParams{}))
	require.Less(t, wantBig, int64(len(big)/bytesPerToken),
		"sanity: the byte-based estimate would have been far larger")

	// A real page-sized JPEG resolves via the JPEG path, and — being well
	// over the pixel budget — tiles from its clamped dimensions, not raw.
	page := encodeJPEG(t, 1700, 2200)
	pw, ph := clampToMaxPixels(1700, 2200, visionParams{}.maxPixels())
	require.Equal(t, int64((pw/visionPatchPixels)*(ph/visionPatchPixels)/visionMergeFactor),
		imageTokensFromBytes(page, visionParams{}))

	// A small image stays under the budget and tiles from raw dimensions.
	small := encodePNG(t, 560, 560) // 0.31M px < budget
	require.Equal(t, int64((560/visionPatchPixels)*(560/visionPatchPixels)/visionMergeFactor),
		imageTokensFromBytes(small, visionParams{}))

	// Unreadable format → single-tile default, not bytes/4.
	require.Equal(t, int64(imageTokensDefault), imageTokensFromBytes(make([]byte, 4000), visionParams{}))

	// Sub-patch image floors instead of returning 0.
	require.Equal(t, int64(imageTokensFloor), imageTokensFromBytes(encodePNG(t, 8, 8), visionParams{}))
}

// The pixel budget is a token ceiling: any page scan larger than the
// budget saturates near visionMaxTokens (llama.cpp's
// set_limit_image_tokens upper bound), and companion-mmproj metadata
// reshapes the tiling for sub-budget images. Guards against the old
// fixed ~1.0M px budget, which capped a 300-DPI page at ~1.3k tokens —
// a 3× prefill under-predict for full-page OCR.
func TestImageTokens_TokenCeilingAndMetadataShape(t *testing.T) {
	// A 300-DPI letter scan (8.4M px) saturates near the token ceiling
	// under the default shape…
	page := encodeJPEG(t, 2550, 3300)
	got := imageTokensFromBytes(page, visionParams{})
	require.Greater(t, got, int64(3500), "large scans must land near the 4096-token ceiling")
	require.LessOrEqual(t, got, int64(visionMaxTokens))

	// …and under the pdf2html mmproj shape (patch 16, merge 2² = 4).
	qwen3 := visionParams{patchPixels: 16, mergeFactor: 4}
	got16 := imageTokensFromBytes(page, qwen3)
	require.Greater(t, got16, int64(3500))
	require.LessOrEqual(t, got16, int64(visionMaxTokens))

	// A sub-budget image tiles exactly by the metadata geometry: the
	// 16-pixel patches of Qwen3-VL yield (16/14)² fewer tokens than the
	// 14-pixel default would claim.
	mid := encodeJPEG(t, 1275, 1650) // 2.1M px, under both budgets
	want := int64((1275 / 16) * (1650 / 16) / 4)
	require.Equal(t, want, imageTokensFromBytes(mid, qwen3))
}

// clampToMaxPixels scales oversized images down (aspect preserved) to the
// encoder's pixel budget and leaves in-budget images untouched. This is the
// single biggest correction to the raw-dimension token estimate: a 300-DPI
// page tiled from raw pixels over-counted tokens by ~9×.
func TestClampToMaxPixels(t *testing.T) {
	tests := []struct {
		name       string
		w, h       int
		wantBudget bool // result must fit within the default pixel budget
		wantSame   bool // result must equal the input (already in budget)
	}{
		{name: "300dpi letter clamps", w: 2550, h: 3300, wantBudget: true},
		{name: "in-budget passes through", w: 560, h: 560, wantSame: true},
		{name: "exactly at budget passes through", w: 1, h: visionParams{}.maxPixels(), wantSame: true},
		{name: "wide aspect preserved", w: 8000, h: 1000, wantBudget: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gw, gh := clampToMaxPixels(tt.w, tt.h, visionParams{}.maxPixels())
			require.GreaterOrEqual(t, gw, 1)
			require.GreaterOrEqual(t, gh, 1)
			if tt.wantSame {
				require.Equal(t, tt.w, gw)
				require.Equal(t, tt.h, gh)
				return
			}
			if tt.wantBudget {
				require.LessOrEqual(t, gw*gh, visionParams{}.maxPixels())
				// Aspect ratio is preserved within rounding.
				require.InDelta(t, float64(tt.w)/float64(tt.h), float64(gw)/float64(gh), 0.05)
			}
		})
	}
}

// predictCost uses the fallback parameter count when the catalogue
// didn't carry a usable number. Lets unknown-size models still produce
// a sensible estimate proportional to a 7B median. Uses an embed job
// to isolate the parameter-count fallback from the chat decode term.
func TestPredictCost_FallbackParameterCount(t *testing.T) {
	job := &llamacpp.Job{
		Kind: llamacpp.JobKind_JOB_KIND_EMBED,
		Body: &llamacpp.Job_Embed{Embed: &llamacpp.EmbedJob{Input: "abcd"}}, // 1 token after /4
	}
	gotCost := predictCost(job, 0, false, visionParams{})
	want := 1.0 * flopsPerTokenPerParam * float64(fallbackParameterCount) / 1e9
	require.InDelta(t, want, gotCost, 1e-6)
}

// Unrecognised job kinds bottom-out at the predictCost one-token floor
// rather than crashing or returning zero — the gateway should never
// emit such a job, but a future kind landed on an old binary must not
// take MASS down. Defensive coverage for the switch's default branch.
func TestPredictCost_UnknownKindFloors(t *testing.T) {
	const params uint64 = 1_000_000_000
	tokenCost := flopsPerTokenPerParam * float64(params) / 1e9
	tests := []struct {
		name string
		job  *llamacpp.Job
	}{
		{
			name: "unspecified kind",
			job:  &llamacpp.Job{Kind: llamacpp.JobKind_JOB_KIND_UNSPECIFIED},
		},
		{
			name: "future enum value",
			job:  &llamacpp.Job{Kind: llamacpp.JobKind(99)},
		},
		{
			// Chat kind, nil Chat body — defensive: matches the
			// chatInputTokens nil-guard.
			name: "chat kind with nil body",
			job:  &llamacpp.Job{Kind: llamacpp.JobKind_JOB_KIND_CHAT},
		},
		{
			// Embed kind, nil embed body.
			name: "embed kind with nil body",
			job:  &llamacpp.Job{Kind: llamacpp.JobKind_JOB_KIND_EMBED},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotCost := predictCost(tt.job, params, false, visionParams{})
			// All four cases floor to a single token (no input, no
			// decode), so cost == tokenCost. (chat-with-nil-body short-
			// circuits at the kind switch on chat helpers, but the
			// outer Job kind isn't CHAT until we touch Body — so
			// jobExpectedDecodeTokens returns 0 here.)
			require.InDelta(t, tokenCost, gotCost, 1e-9)
		})
	}
}

// atoiUint64 is the gateway-private GGUF property parser. Empty
// strings, non-digits, and overflow each have to return 0 so the
// vision-shape lookup degrades to its defaults.
func TestAtoiUint64(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want uint64
	}{
		{"empty", "", 0},
		{"zero", "0", 0},
		{"positive", "42", 42},
		{"larger", "8030261248", 8030261248},
		{"non-digit returns 0", "8B", 0},
		{"leading sign returns 0", "-42", 0},
		{"trailing garbage returns 0", "42x", 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			require.Equal(t, tt.want, atoiUint64(tt.in))
		})
	}
}
