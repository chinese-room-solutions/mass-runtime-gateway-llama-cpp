package gateway

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	gatewaypb "github.com/chinese-room-solutions/mass-proto/gen/go/gateway"
	llamacpp "github.com/chinese-room-solutions/mass-proto/gen/go/llama-cpp"
	workerpb "github.com/chinese-room-solutions/mass-proto/gen/go/worker"
	"github.com/chinese-room-solutions/mass-runtime-gateway-llama-cpp/internal/payload"
	"github.com/rs/zerolog"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// benchModel is one catalogued file for [benchGateway]. Files sharing a
// group are siblings under one operator-typed Name, which is how the
// gateway finds a projector to auto-attach.
type benchModel struct {
	group     string
	filename  string
	modelType string // "chat" / "embedding"; "" for a projector
	companion string // "mmproj" for a projector
}

// benchGateway builds a Gateway over a temp models dir holding the given
// catalogued files. Entries are fully populated (capabilities, both
// parameter counts, mtime/size, hash) so parseModelInfo serves them from
// the catalogue instead of trying to read a GGUF header out of the
// placeholder bytes.
func benchGateway(t *testing.T, models ...benchModel) *Gateway {
	t.Helper()
	tmp := t.TempDir()
	cache := newMetadataCache(tmp, tmp, zerolog.Nop())
	for _, m := range models {
		abs := filepath.Join(tmp, formatDir, m.group, m.filename)
		require.NoError(t, os.MkdirAll(filepath.Dir(abs), 0o755))
		require.NoError(t, os.WriteFile(abs, make([]byte, 4096), 0o644))
		st, err := os.Stat(abs)
		require.NoError(t, err)
		entry := &catalogueEntry{
			Name:                 m.group,
			MTime:                st.ModTime(),
			Size:                 st.Size(),
			Sha256:               "aa",
			ParameterCount:       1_000_000_000,
			ActiveParameterCount: 1_000_000_000,
			ModelType:            m.modelType,
			Companion:            m.companion,
			Capabilities:         &gatewaypb.Capabilities{},
		}
		if m.companion == "mmproj" {
			entry.Properties = map[string]string{"vision_patch_size": "16"}
		}
		cache.entries[formatDir+"/"+m.group+"/"+m.filename] = entry
	}
	h := newHandlers(routerDeps{modelsDir: tmp, cache: cache, logger: zerolog.Nop()})
	return &Gateway{logger: zerolog.Nop(), modelsDir: tmp, cache: cache, h: h}
}

// AuthorBenchPayload authors a real job of the model's own kind, with
// the complete load artifact set, the load hints that job would be run
// under, and the cost the Submit path would put on it. The cost
// assertion is the contract that matters: MASS divides it by measured
// wall time to get the model's throughput, so it has to be the exact
// same unit every later Submit is priced in.
func TestAuthorBenchPayload(t *testing.T) {
	g := benchGateway(t,
		benchModel{group: "chatty", filename: "model.gguf", modelType: "chat"},
		benchModel{group: "embedder", filename: "model.gguf", modelType: "embedding"},
		benchModel{group: "seer", filename: "model.gguf", modelType: "chat"},
		benchModel{group: "seer", filename: "mmproj-seer-f16.gguf", companion: "mmproj"},
	)

	tests := []struct {
		name         string
		modelID      string
		wantJobKind  llamacpp.JobKind
		wantLoadKind llamacpp.LoadKind
		wantFiles    []string // store-relative keys, primary first
		wantMmproj   string   // LoadHints.mmproj_filename
	}{
		{
			name:         "chat model gets a capped chat completion",
			modelID:      "chatty/model.gguf",
			wantJobKind:  llamacpp.JobKind_JOB_KIND_CHAT,
			wantLoadKind: llamacpp.LoadKind_LOAD_KIND_CHAT,
			wantFiles:    []string{"gguf/chatty/model.gguf"},
		},
		{
			name:         "embedding model gets a fixed batch of texts",
			modelID:      "embedder/model.gguf",
			wantJobKind:  llamacpp.JobKind_JOB_KIND_BATCH_EMBED,
			wantLoadKind: llamacpp.LoadKind_LOAD_KIND_EMBEDDING,
			wantFiles:    []string{"gguf/embedder/model.gguf"},
		},
		{
			// The projector is gateway-private knowledge: MASS can't
			// know to ship it unless the response carries it.
			name:         "vision chat model ships its projector too",
			modelID:      "seer/model.gguf",
			wantJobKind:  llamacpp.JobKind_JOB_KIND_CHAT,
			wantLoadKind: llamacpp.LoadKind_LOAD_KIND_CHAT,
			wantFiles:    []string{"gguf/seer/model.gguf", "gguf/seer/mmproj-seer-f16.gguf"},
			wantMmproj:   "mmproj-seer-f16.gguf",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resp, err := g.AuthorBenchPayload(context.Background(), &gatewaypb.AuthorBenchPayloadRequest{ModelId: tt.modelID})
			require.NoError(t, err)
			require.Positive(t, resp.GetCost(), "MASS requires cost > 0")

			job, err := payload.DecodeJob(resp.GetPayload())
			require.NoError(t, err, "payload must decode as an ordinary job")
			require.Equal(t, tt.wantJobKind, job.GetKind())

			hints, err := payload.DecodeLoadHints(resp.GetLoadHints())
			require.NoError(t, err)
			require.Equal(t, tt.wantLoadKind, hints.GetKind())
			require.Equal(t, tt.wantMmproj, hints.GetMmprojFilename())

			// Every file the load needs travels with the response, and
			// the hints' projector is one of them.
			gotFiles := make([]string, 0, len(resp.GetFiles()))
			var mmprojFile *workerpb.ModelFile
			for _, f := range resp.GetFiles() {
				gotFiles = append(gotFiles, f.GetFilename())
				require.NotEmpty(t, f.GetLocalPath())
				require.Equal(t, int64(4096), f.GetSizeBytes())
				if f.GetRole() == workerpb.ModelFileRole_MODEL_FILE_ROLE_MMPROJ {
					mmprojFile = f
				}
			}
			require.Equal(t, tt.wantFiles, gotFiles)
			require.Equal(t, workerpb.ModelFileRole_MODEL_FILE_ROLE_PRIMARY, resp.GetFiles()[0].GetRole())
			if tt.wantMmproj == "" {
				require.Nil(t, mmprojFile)
			} else {
				require.NotNil(t, mmprojFile)
				require.Equal(t, tt.wantMmproj, filepath.Base(mmprojFile.GetFilename()),
					"the shipped projector must be the one load_hints names")
			}

			// The cost MUST equal what the Submit path would price this
			// very payload at — same job, same files, same costing.
			params, err := g.h.buildScheduleParams(context.Background(), tt.modelID, job, hints, resp.GetFiles())
			require.NoError(t, err)
			require.InDelta(t, params.Cost, resp.GetCost(), 1e-9,
				"bench cost and submit cost must be in the same unit")
			require.Equal(t, params.Payload, resp.GetPayload(),
				"bench payload must encode exactly like a submitted job")
			require.Equal(t, params.LoadHints, resp.GetLoadHints())
		})
	}
}

// The bench must be reproducible: the same model authored twice yields
// byte-identical payloads and the same cost, or two benches of one model
// measure different work.
func TestAuthorBenchPayload_Deterministic(t *testing.T) {
	g := benchGateway(t,
		benchModel{group: "chatty", filename: "model.gguf", modelType: "chat"},
		benchModel{group: "embedder", filename: "model.gguf", modelType: "embedding"},
	)
	for _, id := range []string{"chatty/model.gguf", "embedder/model.gguf"} {
		t.Run(id, func(t *testing.T) {
			first, err := g.AuthorBenchPayload(context.Background(), &gatewaypb.AuthorBenchPayloadRequest{ModelId: id})
			require.NoError(t, err)
			second, err := g.AuthorBenchPayload(context.Background(), &gatewaypb.AuthorBenchPayloadRequest{ModelId: id})
			require.NoError(t, err)
			require.Equal(t, first.GetPayload(), second.GetPayload())
			require.Equal(t, first.GetLoadHints(), second.GetLoadHints())
			require.Equal(t, first.GetCost(), second.GetCost())
		})
	}
}

// Bad targets must come back as codes MASS can act on: retryable-looking
// Internal errors would send it into the bench retry loop for a model
// that can never be benched.
func TestAuthorBenchPayload_Errors(t *testing.T) {
	g := benchGateway(t,
		benchModel{group: "chatty", filename: "model.gguf", modelType: "chat"},
		benchModel{group: "projector", filename: "model.gguf", companion: "mmproj"},
	)

	tests := []struct {
		name    string
		modelID string
		want    codes.Code
	}{
		{"empty id", "", codes.InvalidArgument},
		{"blank id", "   ", codes.InvalidArgument},
		{"traversal id", "../etc/passwd", codes.InvalidArgument},
		{"unknown model", "nope/model.gguf", codes.NotFound},
		{"mmproj companion", "projector/model.gguf", codes.InvalidArgument},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := g.AuthorBenchPayload(context.Background(), &gatewaypb.AuthorBenchPayloadRequest{ModelId: tt.modelID})
			require.Error(t, err)
			require.Equal(t, tt.want, status.Code(err))
		})
	}
}

// The chat prompt has to clear predictCost's chatDecodeRatio heuristic,
// otherwise a plain chat model is priced below the token cap while a
// thinking model is priced at it — the same bench payload would carry
// two different costs depending on the model's flags.
func TestBenchChatPromptPricesAtTheCap(t *testing.T) {
	job := benchJob(llamacpp.LoadKind_LOAD_KIND_CHAT)
	const params uint64 = 1_000_000_000
	plain := predictCost(job, params, false, visionParams{})
	thinking := predictCost(job, params, true, visionParams{})
	require.Equal(t, plain, thinking)
}
