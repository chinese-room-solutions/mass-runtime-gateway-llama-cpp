package gateway

import (
	"context"
	"errors"
	"net"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	gatewaypb "github.com/chinese-room-solutions/mass-proto/gen/go/gateway"
	llamacpp "github.com/chinese-room-solutions/mass-proto/gen/go/llama-cpp"
	"github.com/chinese-room-solutions/mass-runtime-gateway-llama-cpp/internal/sched"
	"github.com/rs/zerolog"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/test/bufconn"
)

// dispatchWithID() must call predictCost(job) and pass the resulting (cost,
// axis) on the wire. Tests the integration glue between the estimator
// and the scheduler client.
func TestDispatchPassesCostAndAxisToSubmit(t *testing.T) {
	const promptLen = 400
	job := &llamacpp.Job{
		Kind: llamacpp.JobKind_JOB_KIND_CHAT,
		Body: &llamacpp.Job_Chat{Chat: &llamacpp.ChatJob{
			Messages: []*llamacpp.ChatMessage{
				{Content: strings.Repeat("a", promptLen)},
			},
		}},
	}
	// Test runs without a populated metadata cache, so primaryParameterCount
	// returns 0 and predictCost falls back to fallbackParameterCount. The
	// integration assertion here is that the wire cost equals whatever
	// predictCost would produce given the same fallback path.
	// Without a populated cache, primaryThinking returns false too —
	// dispatch passes thinking=false to predictCost. The integration
	// assertion is that the wire cost equals predictCost for the exact
	// same (job, 0, false) tuple dispatch will compute.
	wantCost, wantAxis := predictCost(job, 0, false, visionParams{})

	fake := &captureScheduler{}
	client, cleanup := newDispatchTestClient(t, fake)
	defer cleanup()

	h := &handlers{scheduler: client, logger: zerolog.Nop()}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	_, chunks, err := h.dispatchWithID(ctx, "model-1", job, nil, nil)
	require.NoError(t, err)
	// Drain the (immediately-terminal) stream so the test doesn't leak goroutines.
	for range chunks {
	}

	require.Equal(t, int32(1), fake.submits.Load(), "dispatch should call Submit exactly once")
	require.InDelta(t, wantCost, fake.lastCost.Load().(float64), 1e-9, "wire cost must equal predictCost(job)")
	require.Equal(t, wantAxis, fake.lastAxis.Load().(string), "wire cost_axis must equal predictCost(job)")
}

// captureScheduler is a minimal MassSchedulerServer that lets dispatchWithID()
// complete without doing real work. Submit records the wire fields;
// StreamChunks emits one Completed frame.
type captureScheduler struct {
	gatewaypb.UnimplementedMassSchedulerServer

	submits      atomic.Int32
	lastCost     atomic.Value // float64
	lastAxis     atomic.Value // string
	lastPriority atomic.Value // gatewaypb.JobPriority
}

func (c *captureScheduler) Submit(_ context.Context, req *gatewaypb.SubmitRequest) (*gatewaypb.SubmitResponse, error) {
	c.submits.Add(1)
	c.lastCost.Store(req.GetCost())
	c.lastAxis.Store(req.GetCostAxis())
	c.lastPriority.Store(req.GetPriority())
	return &gatewaypb.SubmitResponse{JobId: "job-1"}, nil
}

func (c *captureScheduler) StreamChunks(_ *gatewaypb.StreamChunksRequest, s gatewaypb.MassScheduler_StreamChunksServer) error {
	return s.Send(&gatewaypb.ScheduleChunk{
		JobId: "job-1",
		Seq:   0,
		Frame: &gatewaypb.ScheduleChunk_Completed{
			Completed: &gatewaypb.ScheduleCompleted{FinalResponse: nil, WorkerId: "w-1"},
		},
	})
}

func newDispatchTestClient(t *testing.T, server *captureScheduler) (*sched.Client, func()) {
	t.Helper()
	lis := bufconn.Listen(1 << 16)
	grpcServer := grpc.NewServer()
	gatewaypb.RegisterMassSchedulerServer(grpcServer, server)

	var serveErr sync.Once
	go func() {
		if err := grpcServer.Serve(lis); err != nil && !errors.Is(err, grpc.ErrServerStopped) {
			serveErr.Do(func() { t.Logf("grpc.Serve: %v", err) })
		}
	}()

	conn, err := grpc.NewClient(
		"passthrough://bufnet",
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithContextDialer(func(_ context.Context, _ string) (net.Conn, error) {
			return lis.Dial()
		}),
	)
	require.NoError(t, err)

	cleanup := func() {
		_ = conn.Close()
		grpcServer.Stop()
		_ = lis.Close()
	}
	return sched.NewClient(conn, zerolog.Nop()), cleanup
}
