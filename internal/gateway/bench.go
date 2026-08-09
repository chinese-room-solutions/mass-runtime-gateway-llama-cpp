package gateway

import (
	"context"
	"fmt"
	"strings"

	gatewaypb "github.com/chinese-room-solutions/mass-proto/gen/go/gateway"
	llamacpp "github.com/chinese-room-solutions/mass-proto/gen/go/llama-cpp"
	"github.com/chinese-room-solutions/mass-runtime-gateway-llama-cpp/internal/payload"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// Bench payload sizing. MASS runs the payload through the ordinary job
// path on the worker and divides the cost by the measured wall time, so
// the numbers below trade measurement stability against how long an
// operator waits for a fresh model to become schedulable: a bench must
// run in seconds on a CPU-only worker, for a 7B-class model.
const (
	// benchChatMaxTokens is the generation cap for the chat bench.
	// Decode dominates the measurement, so this is the knob that sets
	// the run time: ~64 tokens is a few seconds even on a slow CPU.
	benchChatMaxTokens = 64
	// benchEmbedBatch and benchEmbedTextBytes size the embedding bench.
	// 16 × 512 bytes ≈ 2k prefill tokens — enough work to swamp
	// per-request overhead, short enough to fit any sane context.
	benchEmbedBatch     = 16
	benchEmbedTextBytes = 512
)

// benchChatPrompt is the chat bench's fixed prompt. Two properties
// matter and neither is about the wording: it is long enough (≥ 88
// bytes ≈ 22 tokens) that [predictCost]'s chatDecodeRatio heuristic
// clears benchChatMaxTokens — so plain and thinking models are priced
// at the same cap — and it asks for continuous prose, which keeps a
// model generating up to that cap instead of stopping after a word.
const benchChatPrompt = "Describe in one flowing paragraph of plain prose how a mechanical clock keeps time: the mainspring that stores the energy, the gear train that divides it, the escapement that releases it in equal steps, and the pendulum that sets the beat."

// benchEmbedCorpus is the fixed text the embedding bench's inputs are
// cut from. Pure ASCII so slicing to an exact byte length can never
// split a rune.
const benchEmbedCorpus = "the quick brown fox jumps over the lazy dog while a barge drifts past the quay and the harbour bell counts out the hour in even strokes "

// AuthorBenchPayload writes the request MASS benchmarks model_id with.
// The payload is an ordinary job of this gateway's own encoding — the
// worker runs it through the normal job path — chosen by the model's
// kind: a short capped chat completion for chat models, a fixed batch
// of fixed-length texts for embedding models. Content is fixed, so two
// benches of the same model measure the same work.
//
// files is the complete load artifact set, built exactly as the Submit
// path builds it: the primary plus every companion the gateway attaches
// itself (the sibling mmproj for a vision chat model). The returned
// load_hints name those companions, so shipping anything less than the
// full set fails the load and burns the bench as an incapable verdict.
//
// cost comes from [handlers.jobCost], the same costing the Submit path
// puts on every job, which is what makes MASS's `cost / elapsed`
// division meaningful for later submits of this model.
//
// Returns NotFound when no catalogued model matches model_id,
// InvalidArgument when the id is malformed or names a projector
// (nothing to run on its own), and FailedPrecondition before Init.
func (g *Gateway) AuthorBenchPayload(ctx context.Context, req *gatewaypb.AuthorBenchPayloadRequest) (*gatewaypb.AuthorBenchPayloadResponse, error) {
	_ = ctx
	h, ok := g.benchHandlers()
	if !ok {
		return nil, errNotInitialised("author_bench_payload")
	}
	id := strings.TrimSpace(req.GetModelId())
	if id == "" {
		return nil, status.Error(codes.InvalidArgument, "author_bench_payload: model_id is required")
	}
	storePath, err := h.resolveStorePath(id)
	if err != nil {
		return nil, status.Errorf(codes.InvalidArgument, "author_bench_payload: %v", err)
	}
	info, ok := h.cache.parseModelInfo(absForStoreID(h.modelsDir, storePath), storePath)
	if !ok {
		return nil, status.Errorf(codes.NotFound, "author_bench_payload: no model matches id %q", id)
	}
	if info.Companion != "" {
		return nil, status.Errorf(codes.InvalidArgument,
			"author_bench_payload: %q is a %s companion, not a model that runs on its own", id, info.Companion)
	}

	loadKind := llamacpp.LoadKind_LOAD_KIND_CHAT
	if info.ModelType == "embedding" {
		loadKind = llamacpp.LoadKind_LOAD_KIND_EMBEDDING
	}
	hints, files, err := h.buildLoadArtifacts(storePath, nil, loadKind)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "author_bench_payload: build load artifacts: %v", err)
	}
	job := benchJob(loadKind)
	payloadBytes, err := payload.EncodeJob(job)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "author_bench_payload: encode payload: %v", err)
	}
	hintsBytes, err := payload.EncodeLoadHints(hints)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "author_bench_payload: encode load hints: %v", err)
	}
	return &gatewaypb.AuthorBenchPayloadResponse{
		Payload:   payloadBytes,
		LoadHints: hintsBytes,
		Files:     files,
		Cost:      h.jobCost(job, files),
	}, nil
}

// benchHandlers returns the Init-populated handlers. ok=false until Init
// has run — callers translate that to FailedPrecondition rather than
// dereferencing nil.
func (g *Gateway) benchHandlers() (*handlers, bool) {
	g.mu.RLock()
	defer g.mu.RUnlock()
	return g.h, g.h != nil && g.h.cache != nil
}

// benchJob builds the representative job for a load kind. Sampling is
// pinned (greedy, fixed seed, hard token cap) so the run is
// reproducible and bounded; the chat bench measures decode, the embed
// bench measures prefill.
func benchJob(kind llamacpp.LoadKind) *llamacpp.Job {
	if kind == llamacpp.LoadKind_LOAD_KIND_EMBEDDING {
		return &llamacpp.Job{
			Kind: llamacpp.JobKind_JOB_KIND_BATCH_EMBED,
			Body: &llamacpp.Job_BatchEmbed{BatchEmbed: &llamacpp.BatchEmbedJob{Inputs: benchEmbedInputs()}},
		}
	}
	maxTokens, temperature, seed := int32(benchChatMaxTokens), float32(0), int32(0)
	return &llamacpp.Job{
		Kind: llamacpp.JobKind_JOB_KIND_CHAT,
		Body: &llamacpp.Job_Chat{Chat: &llamacpp.ChatJob{
			Messages: []*llamacpp.ChatMessage{{
				Role:    llamacpp.Role_ROLE_USER,
				Content: benchChatPrompt,
			}},
			Sampling: &llamacpp.SamplingParams{
				MaxTokens:   &maxTokens,
				Temperature: &temperature,
				Seed:        &seed,
			},
		}},
	}
}

// benchEmbedInputs returns the embedding bench's batch: benchEmbedBatch
// distinct texts of exactly benchEmbedTextBytes bytes each, cut from a
// fixed corpus and numbered so no two inputs are identical.
func benchEmbedInputs() []string {
	filler := strings.Repeat(benchEmbedCorpus, 1+benchEmbedTextBytes/len(benchEmbedCorpus))
	out := make([]string, benchEmbedBatch)
	for i := range out {
		head := fmt.Sprintf("%02d ", i)
		out[i] = head + filler[:benchEmbedTextBytes-len(head)]
	}
	return out
}
