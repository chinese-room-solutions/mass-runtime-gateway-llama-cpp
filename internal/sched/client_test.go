package sched

import (
	"context"
	"errors"
	"net"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	gatewaypb "github.com/chinese-room-solutions/mass-proto/gen/go/gateway"
	"github.com/rs/zerolog"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
	"google.golang.org/grpc/test/bufconn"
)

// Client.Schedule splits into Submit+StreamChunks; a transient stream
// error on the first StreamChunks call must trigger exactly one reconnect
// using resume_seq=last_seen_seq+1, and the second call's frames must be
// concatenated to the first call's frames into one ordered out-channel.
func TestClient_ScheduleResumesOnceAcrossStreamError(t *testing.T) {
	fake := &fakeMassScheduler{
		streamHandlers: []streamHandler{
			// First call: send seq=0, then error mid-stream.
			func(req *gatewaypb.StreamChunksRequest, s gatewaypb.MassScheduler_StreamChunksServer) error {
				require.Equal(t, uint64(0), req.GetResumeSeq(), "first attach should request seq=0")
				if err := s.Send(bodyChunk(req.GetJobId(), 0, "alpha")); err != nil {
					return err
				}
				return status.Error(codes.Unavailable, "simulated transient drop")
			},
			// Second call: resume from seq=1, deliver one body chunk + terminal.
			func(req *gatewaypb.StreamChunksRequest, s gatewaypb.MassScheduler_StreamChunksServer) error {
				require.Equal(t, uint64(1), req.GetResumeSeq(), "second attach should request seq=1")
				if err := s.Send(bodyChunk(req.GetJobId(), 1, "beta")); err != nil {
					return err
				}
				return s.Send(&gatewaypb.ScheduleChunk{
					JobId: req.GetJobId(),
					Seq:   2,
					Frame: &gatewaypb.ScheduleChunk_Completed{
						Completed: &gatewaypb.ScheduleCompleted{
							FinalResponse: []byte("done"),
							WorkerId:      "worker-1",
						},
					},
				})
			},
		},
	}

	client, cleanup := newTestClient(t, fake)
	defer cleanup()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	out, err := client.Schedule(ctx, ScheduleParams{ModelID: "m1", Payload: []byte("hi")})
	require.NoError(t, err)

	got := drainChunks(t, out)
	require.Len(t, got, 3)
	require.Equal(t, ChunkBody, got[0].Type)
	require.Equal(t, []byte("alpha"), got[0].Chunk)
	require.Equal(t, ChunkBody, got[1].Type)
	require.Equal(t, []byte("beta"), got[1].Chunk)
	require.Equal(t, ChunkCompleted, got[2].Type)
	require.Equal(t, []byte("done"), got[2].Final)
	require.Equal(t, "worker-1", got[2].WorkerID)
	require.Equal(t, int32(2), fake.streamCalls.Load(), "expected exactly one reconnect")
}

// MASS always ends a stream with a terminal frame, so a clean EOF without one
// means the stream was cut short while the job is still durable on the MASS
// side. That must resume from lastSeq+1 like any other break — by seq only,
// with no re-submission and no re-run of the inference.
func TestClient_ScheduleResumesTruncatedStream(t *testing.T) {
	fake := &fakeMassScheduler{
		streamHandlers: []streamHandler{
			// First call: one body chunk, then return cleanly — EOF, no terminal.
			func(req *gatewaypb.StreamChunksRequest, s gatewaypb.MassScheduler_StreamChunksServer) error {
				require.Equal(t, uint64(0), req.GetResumeSeq(), "first attach should request seq=0")
				return s.Send(bodyChunk(req.GetJobId(), 0, "alpha"))
			},
			// Second call: resume after the last frame we saw and finish.
			func(req *gatewaypb.StreamChunksRequest, s gatewaypb.MassScheduler_StreamChunksServer) error {
				require.Equal(t, uint64(1), req.GetResumeSeq(), "resume must continue from lastSeq+1")
				if err := s.Send(bodyChunk(req.GetJobId(), 1, "beta")); err != nil {
					return err
				}
				return s.Send(&gatewaypb.ScheduleChunk{
					JobId: req.GetJobId(),
					Seq:   2,
					Frame: &gatewaypb.ScheduleChunk_Completed{
						Completed: &gatewaypb.ScheduleCompleted{
							FinalResponse: []byte("done"),
							WorkerId:      "worker-1",
						},
					},
				})
			},
		},
	}

	client, cleanup := newTestClient(t, fake)
	defer cleanup()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	out, err := client.Schedule(ctx, ScheduleParams{ModelID: "m1", Payload: []byte("hi")})
	require.NoError(t, err)

	got := drainChunks(t, out)
	require.Len(t, got, 3, "a truncated stream must be resumed, not dropped")
	require.Equal(t, ChunkBody, got[0].Type)
	require.Equal(t, []byte("alpha"), got[0].Chunk)
	require.Equal(t, ChunkBody, got[1].Type)
	require.Equal(t, []byte("beta"), got[1].Chunk)
	require.Equal(t, ChunkCompleted, got[2].Type)
	require.Equal(t, []byte("done"), got[2].Final)
	require.Equal(t, int32(2), fake.streamCalls.Load(), "expected exactly one reconnect")
	require.Equal(t, int32(1), fake.submitCalls.Load(), "resume must not re-submit the job")
}

// Truncation twice over exhausts the one allowed retry and surfaces a
// ChunkError rather than reconnecting forever.
func TestClient_ScheduleGivesUpOnRepeatedTruncation(t *testing.T) {
	noTerminal := func(_ *gatewaypb.StreamChunksRequest, _ gatewaypb.MassScheduler_StreamChunksServer) error {
		return nil
	}
	fake := &fakeMassScheduler{streamHandlers: []streamHandler{noTerminal, noTerminal}}

	client, cleanup := newTestClient(t, fake)
	defer cleanup()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	out, err := client.Schedule(ctx, ScheduleParams{ModelID: "m1"})
	require.NoError(t, err)

	got := drainChunks(t, out)
	require.Len(t, got, 1)
	require.Equal(t, ChunkError, got[0].Type)
	require.Equal(t, ErrStreamTruncated.Error(), got[0].ErrText)
	require.Equal(t, int32(2), fake.streamCalls.Load(), "client must stop after the one retry")
	require.Equal(t, int32(1), fake.submitCalls.Load(), "giving up must not re-submit the job")
}

// A second stream error after the one allowed retry must surface as
// ChunkError, not trigger another reconnect.
func TestClient_ScheduleGivesUpAfterOneRetry(t *testing.T) {
	fake := &fakeMassScheduler{
		streamHandlers: []streamHandler{
			func(_ *gatewaypb.StreamChunksRequest, _ gatewaypb.MassScheduler_StreamChunksServer) error {
				return status.Error(codes.Unavailable, "first drop")
			},
			func(_ *gatewaypb.StreamChunksRequest, _ gatewaypb.MassScheduler_StreamChunksServer) error {
				return status.Error(codes.Unavailable, "second drop")
			},
		},
	}

	client, cleanup := newTestClient(t, fake)
	defer cleanup()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	out, err := client.Schedule(ctx, ScheduleParams{ModelID: "m1"})
	require.NoError(t, err)

	got := drainChunks(t, out)
	require.Len(t, got, 1)
	require.Equal(t, ChunkError, got[0].Type)
	require.Equal(t, int32(2), fake.streamCalls.Load(), "client must stop after the one retry")
}

// ScheduleWithID returns the Submit-assigned job id alongside the stream, so a
// handler can surface it (X-Mass-Job-Id) before the body streams.
func TestClient_ScheduleWithIDReturnsJobIDAndStreams(t *testing.T) {
	fake := &fakeMassScheduler{
		streamHandlers: []streamHandler{
			func(req *gatewaypb.StreamChunksRequest, s gatewaypb.MassScheduler_StreamChunksServer) error {
				require.Equal(t, "job-1", req.GetJobId(), "stream must target the submitted job")
				return s.Send(&gatewaypb.ScheduleChunk{
					JobId: req.GetJobId(),
					Frame: &gatewaypb.ScheduleChunk_Completed{
						Completed: &gatewaypb.ScheduleCompleted{FinalResponse: []byte("done")},
					},
				})
			},
		},
	}
	client, cleanup := newTestClient(t, fake)
	defer cleanup()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	jobID, out, err := client.ScheduleWithID(ctx, ScheduleParams{ModelID: "m1"})
	require.NoError(t, err)
	require.Equal(t, "job-1", jobID)
	require.Equal(t, int32(1), fake.submitCalls.Load(), "fresh schedule submits once")

	got := drainChunks(t, out)
	require.Len(t, got, 1)
	require.Equal(t, ChunkCompleted, got[0].Type)
}

// Reattach streams an existing job by id WITHOUT submitting a new one, asking
// for resume_seq=0 so MASS replays buffered/stored frames then live ones. This
// is the resume-after-disconnect path.
func TestClient_ReattachStreamsWithoutSubmit(t *testing.T) {
	fake := &fakeMassScheduler{
		streamHandlers: []streamHandler{
			func(req *gatewaypb.StreamChunksRequest, s gatewaypb.MassScheduler_StreamChunksServer) error {
				require.Equal(t, "existing-job", req.GetJobId())
				require.Equal(t, uint64(0), req.GetResumeSeq(), "reattach replays from seq 0")
				if err := s.Send(bodyChunk(req.GetJobId(), 0, "replayed")); err != nil {
					return err
				}
				return s.Send(&gatewaypb.ScheduleChunk{
					JobId: req.GetJobId(),
					Seq:   1,
					Frame: &gatewaypb.ScheduleChunk_Completed{
						Completed: &gatewaypb.ScheduleCompleted{FinalResponse: []byte("done")},
					},
				})
			},
		},
	}
	client, cleanup := newTestClient(t, fake)
	defer cleanup()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	got := drainChunks(t, client.Reattach(ctx, "existing-job"))

	require.Equal(t, int32(0), fake.submitCalls.Load(), "reattach must not submit a new job")
	require.Len(t, got, 2)
	require.Equal(t, []byte("replayed"), got[0].Chunk)
	require.Equal(t, ChunkCompleted, got[1].Type)
}

// --- fake MassScheduler over bufconn ---

type streamHandler func(*gatewaypb.StreamChunksRequest, gatewaypb.MassScheduler_StreamChunksServer) error

type fakeMassScheduler struct {
	gatewaypb.UnimplementedMassSchedulerServer

	streamHandlers []streamHandler
	streamCalls    atomic.Int32
	submitCalls    atomic.Int32

	getResultFn func(*gatewaypb.GetResultRequest) (*gatewaypb.GetResultResponse, error)
	cancelJobFn func(*gatewaypb.CancelJobRequest) (*gatewaypb.CancelJobResponse, error)
}

func (f *fakeMassScheduler) GetResult(_ context.Context, req *gatewaypb.GetResultRequest) (*gatewaypb.GetResultResponse, error) {
	if f.getResultFn == nil {
		return nil, status.Error(codes.Unimplemented, "getResultFn not set")
	}
	return f.getResultFn(req)
}

func (f *fakeMassScheduler) CancelJob(_ context.Context, req *gatewaypb.CancelJobRequest) (*gatewaypb.CancelJobResponse, error) {
	if f.cancelJobFn == nil {
		return nil, status.Error(codes.Unimplemented, "cancelJobFn not set")
	}
	return f.cancelJobFn(req)
}

func (f *fakeMassScheduler) Submit(_ context.Context, _ *gatewaypb.SubmitRequest) (*gatewaypb.SubmitResponse, error) {
	f.submitCalls.Add(1)
	return &gatewaypb.SubmitResponse{JobId: "job-1"}, nil
}

func (f *fakeMassScheduler) StreamChunks(req *gatewaypb.StreamChunksRequest, s gatewaypb.MassScheduler_StreamChunksServer) error {
	idx := int(f.streamCalls.Add(1) - 1)
	if idx >= len(f.streamHandlers) {
		return status.Error(codes.Internal, "no more stream handlers configured")
	}
	return f.streamHandlers[idx](req, s)
}

func newTestClient(t *testing.T, server *fakeMassScheduler) (*Client, func()) {
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
	return NewClient(conn, zerolog.Nop()), cleanup
}

func bodyChunk(jobID string, seq uint64, body string) *gatewaypb.ScheduleChunk {
	return &gatewaypb.ScheduleChunk{
		JobId: jobID,
		Seq:   seq,
		Frame: &gatewaypb.ScheduleChunk_Chunk{Chunk: []byte(body)},
	}
}

func drainChunks(t *testing.T, ch <-chan JobChunk) []JobChunk {
	t.Helper()
	var out []JobChunk
	timeout := time.After(5 * time.Second)
	for {
		select {
		case c, ok := <-ch:
			if !ok {
				return out
			}
			out = append(out, c)
		case <-timeout:
			t.Fatalf("drainChunks: timeout after %d chunks", len(out))
		}
	}
}
