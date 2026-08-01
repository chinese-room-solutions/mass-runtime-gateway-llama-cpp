package gateway

import (
	"context"
	"os"
	"path/filepath"
	"sync"
	"testing"

	gatewaypb "github.com/chinese-room-solutions/mass-proto/gen/go/gateway"
	workerpb "github.com/chinese-room-solutions/mass-proto/gen/go/worker"
	"github.com/rs/zerolog"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// PlanDelete resolves a model id (Group.id slug) to every store-relative
// file sharing the model's Name — primary + mmproj companion — so MASS can
// remove them. The gateway only decides; it never touches the store.
func TestPlanDelete(t *testing.T) {
	modelsDir := t.TempDir()
	slugDir := filepath.Join(formatRoot(modelsDir), "pdf2doc")
	require.NoError(t, os.MkdirAll(slugDir, 0o755))
	primary := filepath.Join(slugDir, "pdf2doc-Q4_K_M.gguf")
	mmproj := filepath.Join(slugDir, "mmproj-pdf2doc-f16.gguf")
	require.NoError(t, os.WriteFile(primary, make([]byte, 16), 0o644))
	require.NoError(t, os.WriteFile(mmproj, make([]byte, 16), 0o644))

	cache := &metadataCache{modelsDir: modelsDir, entries: map[string]*catalogueEntry{
		"gguf/pdf2doc/pdf2doc-Q4_K_M.gguf":     {Name: "pdf2doc"},
		"gguf/pdf2doc/mmproj-pdf2doc-f16.gguf": {Name: "pdf2doc", Companion: "mmproj"},
	}}
	g := &Gateway{modelsDir: modelsDir, cache: cache, logger: zerolog.Nop()}

	want := []string{"gguf/pdf2doc/pdf2doc-Q4_K_M.gguf", "gguf/pdf2doc/mmproj-pdf2doc-f16.gguf"}

	t.Run("resolves primary + companion by group slug", func(t *testing.T) {
		resp, err := g.PlanDelete(context.Background(), &gatewaypb.PlanDeleteRequest{Id: modelSlug("pdf2doc")})
		require.NoError(t, err)
		require.ElementsMatch(t, want, resp.GetRelPaths())
		// Decision only: files must still be on disk.
		require.FileExists(t, primary)
		require.FileExists(t, mmproj)
	})

	t.Run("resolves whole model from a single file id", func(t *testing.T) {
		// The dashboard passes a per-file Model.id (relpath); deleting it
		// must still pull the companion so the whole model goes.
		resp, err := g.PlanDelete(context.Background(), &gatewaypb.PlanDeleteRequest{Id: "pdf2doc/pdf2doc-Q4_K_M.gguf"})
		require.NoError(t, err)
		require.ElementsMatch(t, want, resp.GetRelPaths())
	})

	t.Run("unknown id -> NotFound", func(t *testing.T) {
		_, err := g.PlanDelete(context.Background(), &gatewaypb.PlanDeleteRequest{Id: "no-such-model"})
		require.Equal(t, codes.NotFound, status.Code(err))
	})

	t.Run("already-deleted published id -> empty plan, no error", func(t *testing.T) {
		// A group's variants delete one id at a time; the first delete
		// removes every file sharing the Name, so a later id resolves to
		// nothing. A published-id-shaped target that's already gone is an
		// idempotent no-op, not a NotFound.
		resp, err := g.PlanDelete(context.Background(), &gatewaypb.PlanDeleteRequest{Id: "pdf2doc/mmproj-already-gone.gguf"})
		require.NoError(t, err)
		require.Empty(t, resp.GetRelPaths())
	})

	t.Run("empty id -> InvalidArgument", func(t *testing.T) {
		_, err := g.PlanDelete(context.Background(), &gatewaypb.PlanDeleteRequest{})
		require.Equal(t, codes.InvalidArgument, status.Code(err))
	})
}

// loadedInstancesForRelPaths matches a doomed group's FULL store-relative
// keys against the workers' loaded instances. It prefers matching the
// worker-reported LoadedModelStatus.files (same key namespace as relPaths,
// gguf-prefixed) by exact-or-directory-subtree overlap, and falls back to
// parsing the model_id relpath (which omits the runtime-owned first
// segment) only when a worker reports no files. Every matching instance
// across every worker is returned exactly once, carrying its active count.
func TestLoadedInstancesForRelPaths(t *testing.T) {
	// worker builds a summary of loaded models. Each spec is
	// "model_id | comma,sep,files | active"; files/active are optional.
	worker := func(models ...*workerpb.LoadedModelStatus) *gatewaypb.WorkerSummary {
		return &gatewaypb.WorkerSummary{LoadedModels: models}
	}
	lm := func(id string, active int32, files ...string) *workerpb.LoadedModelStatus {
		return &workerpb.LoadedModelStatus{ModelId: id, Active: active, Files: files}
	}

	tests := []struct {
		name     string
		workers  []*gatewaypb.WorkerSummary
		relPaths []string
		want     []loadedInstance
	}{
		{
			name:     "no workers",
			relPaths: []string{"gguf/pdf2doc/pdf2doc-Q4.gguf"},
			want:     nil,
		},
		{
			name:     "nothing loaded",
			workers:  []*gatewaypb.WorkerSummary{worker()},
			relPaths: []string{"gguf/pdf2doc/pdf2doc-Q4.gguf"},
			want:     nil,
		},
		{
			name: "files-first exact match, hash-carrying id preserved",
			workers: []*gatewaypb.WorkerSummary{worker(
				lm("pdf2doc/pdf2doc-Q4.gguf#abc123", 0, "gguf/pdf2doc/pdf2doc-Q4.gguf"),
				lm("other/other.gguf#dead00", 0, "gguf/other/other.gguf"),
			)},
			relPaths: []string{"gguf/pdf2doc/pdf2doc-Q4.gguf"},
			want:     []loadedInstance{{modelID: "pdf2doc/pdf2doc-Q4.gguf#abc123"}},
		},
		{
			name: "files-first carries active count",
			workers: []*gatewaypb.WorkerSummary{worker(
				lm("pdf2doc/pdf2doc-Q4.gguf#abc123", 3, "gguf/pdf2doc/pdf2doc-Q4.gguf"),
			)},
			relPaths: []string{"gguf/pdf2doc/pdf2doc-Q4.gguf"},
			want:     []loadedInstance{{modelID: "pdf2doc/pdf2doc-Q4.gguf#abc123", active: 3}},
		},
		{
			name: "directory-subtree overlap: relPath is an ancestor of the reported file",
			workers: []*gatewaypb.WorkerSummary{worker(
				lm("some-model#abc123", 0, "onnx/some-model/weights.onnx"),
			)},
			relPaths: []string{"onnx/some-model"},
			want:     []loadedInstance{{modelID: "some-model#abc123"}},
		},
		{
			name: "directory-subtree overlap: reported file is an ancestor of the relPath",
			workers: []*gatewaypb.WorkerSummary{worker(
				lm("some-model#abc123", 0, "onnx/some-model"),
			)},
			relPaths: []string{"onnx/some-model/weights.onnx"},
			want:     []loadedInstance{{modelID: "some-model#abc123"}},
		},
		{
			name: "fallback to model_id relpath when worker reports no files",
			workers: []*gatewaypb.WorkerSummary{worker(
				lm("pdf2doc/pdf2doc-Q4.gguf#abc123", 2),
			)},
			relPaths: []string{"gguf/pdf2doc/pdf2doc-Q4.gguf"},
			want:     []loadedInstance{{modelID: "pdf2doc/pdf2doc-Q4.gguf#abc123", active: 2}},
		},
		{
			name: "fallback: hashless model_id matches too",
			workers: []*gatewaypb.WorkerSummary{worker(
				lm("pdf2doc/pdf2doc-Q4.gguf", 0),
			)},
			relPaths: []string{"gguf/pdf2doc/pdf2doc-Q4.gguf"},
			want:     []loadedInstance{{modelID: "pdf2doc/pdf2doc-Q4.gguf"}},
		},
		{
			name: "distinct load configs of one file are all matched",
			workers: []*gatewaypb.WorkerSummary{worker(
				lm("pdf2doc/pdf2doc-Q4.gguf#abc123", 0, "gguf/pdf2doc/pdf2doc-Q4.gguf"),
				lm("pdf2doc/pdf2doc-Q4.gguf#def456", 0, "gguf/pdf2doc/pdf2doc-Q4.gguf"),
			)},
			relPaths: []string{"gguf/pdf2doc/pdf2doc-Q4.gguf"},
			want: []loadedInstance{
				{modelID: "pdf2doc/pdf2doc-Q4.gguf#abc123"},
				{modelID: "pdf2doc/pdf2doc-Q4.gguf#def456"},
			},
		},
		{
			name: "same instance on two workers is deduplicated",
			workers: []*gatewaypb.WorkerSummary{
				worker(lm("pdf2doc/pdf2doc-Q4.gguf#abc123", 0, "gguf/pdf2doc/pdf2doc-Q4.gguf")),
				worker(lm("pdf2doc/pdf2doc-Q4.gguf#abc123", 0, "gguf/pdf2doc/pdf2doc-Q4.gguf")),
			},
			relPaths: []string{"gguf/pdf2doc/pdf2doc-Q4.gguf"},
			want:     []loadedInstance{{modelID: "pdf2doc/pdf2doc-Q4.gguf#abc123"}},
		},
		{
			name: "companion files in the group match via reported files",
			workers: []*gatewaypb.WorkerSummary{worker(
				lm("pdf2doc/pdf2doc-Q4.gguf#abc123", 0, "gguf/pdf2doc/pdf2doc-Q4.gguf", "gguf/pdf2doc/mmproj-f16.gguf"),
			)},
			relPaths: []string{"gguf/pdf2doc/pdf2doc-Q4.gguf", "gguf/pdf2doc/mmproj-f16.gguf"},
			want:     []loadedInstance{{modelID: "pdf2doc/pdf2doc-Q4.gguf#abc123"}},
		},
		{
			name: "unrelated group untouched",
			workers: []*gatewaypb.WorkerSummary{worker(
				lm("other/other.gguf#dead00", 0, "gguf/other/other.gguf"),
			)},
			relPaths: []string{"gguf/pdf2doc/pdf2doc-Q4.gguf"},
			want:     nil,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			require.Equal(t, tt.want, loadedInstancesForRelPaths(tt.workers, tt.relPaths))
		})
	}
}

// planDeleteScheduler is a fake MassScheduler exposing ListWorkers +
// EvictModel so PlanDelete's residency gate and idle-eviction path can be
// exercised end-to-end through the real sched.Client.
type planDeleteScheduler struct {
	gatewaypb.UnimplementedMassSchedulerServer
	workers []*gatewaypb.WorkerSummary

	mu       sync.Mutex
	evicted  []string
	evictErr error
}

func (s *planDeleteScheduler) ListWorkers(context.Context, *gatewaypb.ListWorkersRequest) (*gatewaypb.ListWorkersResponse, error) {
	return &gatewaypb.ListWorkersResponse{Workers: s.workers}, nil
}

func (s *planDeleteScheduler) EvictModel(_ context.Context, req *gatewaypb.EvictModelRequest) (*gatewaypb.EvictModelResponse, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.evictErr != nil {
		return nil, s.evictErr
	}
	s.evicted = append(s.evicted, req.GetModelId())
	return &gatewaypb.EvictModelResponse{EvictedCount: 1}, nil
}

// PlanDelete refuses when a matching instance is serving, and evicts idle
// residents before handing back the plan. Residency is discovered through
// the real sched.Client against a fake MassScheduler.
func TestPlanDelete_Residency(t *testing.T) {
	newGateway := func(t *testing.T, sched *planDeleteScheduler) (*Gateway, []string) {
		t.Helper()
		modelsDir := t.TempDir()
		slugDir := filepath.Join(formatRoot(modelsDir), "pdf2doc")
		require.NoError(t, os.MkdirAll(slugDir, 0o755))
		require.NoError(t, os.WriteFile(filepath.Join(slugDir, "pdf2doc-Q4.gguf"), make([]byte, 16), 0o644))
		cache := &metadataCache{modelsDir: modelsDir, entries: map[string]*catalogueEntry{
			"gguf/pdf2doc/pdf2doc-Q4.gguf": {Name: "pdf2doc"},
		}}
		client, cleanup := newBatchChatTestClient(t, sched)
		t.Cleanup(cleanup)
		g := &Gateway{modelsDir: modelsDir, cache: cache, scheduler: client, logger: zerolog.Nop()}
		return g, []string{"gguf/pdf2doc/pdf2doc-Q4.gguf"}
	}

	loaded := func(active int32, files ...string) []*gatewaypb.WorkerSummary {
		return []*gatewaypb.WorkerSummary{{LoadedModels: []*workerpb.LoadedModelStatus{
			{ModelId: "pdf2doc/pdf2doc-Q4.gguf#abc123", Active: active, Files: files},
		}}}
	}

	t.Run("active instance -> FailedPrecondition, nothing evicted", func(t *testing.T) {
		sched := &planDeleteScheduler{workers: loaded(2, "gguf/pdf2doc/pdf2doc-Q4.gguf")}
		g, _ := newGateway(t, sched)
		_, err := g.PlanDelete(context.Background(), &gatewaypb.PlanDeleteRequest{Id: modelSlug("pdf2doc")})
		require.Equal(t, codes.FailedPrecondition, status.Code(err))
		require.Empty(t, sched.evicted)
	})

	t.Run("idle instance -> evicted, plan returned", func(t *testing.T) {
		sched := &planDeleteScheduler{workers: loaded(0, "gguf/pdf2doc/pdf2doc-Q4.gguf")}
		g, want := newGateway(t, sched)
		resp, err := g.PlanDelete(context.Background(), &gatewaypb.PlanDeleteRequest{Id: modelSlug("pdf2doc")})
		require.NoError(t, err)
		require.Equal(t, want, resp.GetRelPaths())
		require.Equal(t, []string{"pdf2doc/pdf2doc-Q4.gguf#abc123"}, sched.evicted)
	})

	t.Run("no residents -> plan returned, no eviction", func(t *testing.T) {
		sched := &planDeleteScheduler{}
		g, want := newGateway(t, sched)
		resp, err := g.PlanDelete(context.Background(), &gatewaypb.PlanDeleteRequest{Id: modelSlug("pdf2doc")})
		require.NoError(t, err)
		require.Equal(t, want, resp.GetRelPaths())
		require.Empty(t, sched.evicted)
	})
}
