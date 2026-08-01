// Package sched is a thin wrapper around the gateway-side MassScheduler
// gRPC client. It exists so handlers can call sched.Schedule(ctx, ...)
// without dragging the proto types through every callsite, and so we have
// one place to translate between gateway-internal "Job" types and the
// MassScheduler wire shape.
package sched

import (
	"context"
	"errors"
	"fmt"
	"io"

	"github.com/KernelPryanic/ctxerr"
	gatewaypb "github.com/chinese-room-solutions/mass-proto/gen/go/gateway"
	workerpb "github.com/chinese-room-solutions/mass-proto/gen/go/worker"
	"github.com/rs/zerolog"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// Client wraps the MassScheduler gRPC client.
type Client struct {
	c      gatewaypb.MassSchedulerClient
	logger zerolog.Logger
}

// NewClient builds a wrapper around the brokered MASS connection. Pass the
// connection returned by [plugin.GRPCBroker.Dial].
func NewClient(conn *grpc.ClientConn, logger zerolog.Logger) *Client {
	return &Client{c: gatewaypb.NewMassSchedulerClient(conn), logger: logger.With().Str("component", "sched_client").Logger()}
}

// JobChunk is the shape we hand back to handlers — one streaming frame from
// the worker. Final chunk has Type == ChunkCompleted (Final may be empty
// if the worker streamed everything).
type JobChunk struct {
	Type         ChunkType
	Chunk        []byte
	Final        []byte
	ProgressPct  float32
	ProgressNote string
	ErrText      string
	WorkerID     string
}

// ChunkType discriminates JobChunk variants.
type ChunkType int

const (
	ChunkBody ChunkType = iota
	ChunkProgress
	ChunkCompleted
	ChunkError
)

// ScheduleParams is the gateway-facing argument set for Schedule. Mapping
// to gatewaypb.SubmitRequest is mechanical. Files + LoadHints travel with
// every Submit so MASS can load-on-demand at dispatch when the chosen
// worker doesn't already have the model resident.
//
// Cost + CostAxis are the gateway's prediction of this job's compute
// work on the runtime's reference workload (see internal/gateway/weight.go
// — all prediction physics lives there, on the runtime side). MASS never
// interprets the units: the only quantity it derives is time, by dividing
// Cost by the chosen worker's benched throughput on CostAxis. Units are
// runtime-private; MASS only compares within a runtime.
type ScheduleParams struct {
	ModelID   string
	Payload   []byte
	Cost      float64
	CostAxis  string
	Files     []*workerpb.ModelFile
	LoadHints []byte
	// BaseLoadBytes is the gateway's prediction of the fixed
	// device-memory cost the load pays regardless of concurrency
	// (weights + scratch). MASS uses it to filter workers whose free
	// memory can't fit and to anchor the pool-size projection. 0 =
	// unknown — MASS skips the eligibility check.
	BaseLoadBytes int64
	// PerSlotBytes is the gateway's prediction of the per-slot
	// incremental cost (KV at the configured ctx). MASS combines it
	// with the chosen worker's free memory to project the
	// post-headroom pool size used for wall-clock load latency. 0 =
	// no concurrency dimension — projection collapses to pool=1.
	PerSlotBytes int64
	// HeadroomPct is the device-memory watermark (1-100) the worker
	// will respect when growing the pool. 0 = unknown, MASS falls
	// back to a runtime-agnostic constant.
	HeadroomPct int32
	Source      string
	// Priority orders the job within the chosen worker's queue. Zero
	// (UNSPECIFIED) is treated as MEDIUM on the MASS side.
	Priority gatewaypb.JobPriority
}

// Schedule submits a job and returns a channel of streamed chunks. The
// channel closes after a Completed or Error frame (or on context cancel).
//
// Internally Schedule performs a Submit followed by StreamChunks. If the
// MASS-side chunk stream breaks mid-job — a transient gRPC disconnect, a
// gateway-side network blip, or a clean EOF with no terminal frame —
// Schedule reconnects to StreamChunks once using the last observed seq + 1
// so the gateway resumes where it left off. The job is never re-submitted
// and the inference never re-runs: it stays durable on the MASS side and
// only the missing frames are re-read.
func (c *Client) Schedule(ctx context.Context, p ScheduleParams) (<-chan JobChunk, error) {
	_, chunks, err := c.ScheduleWithID(ctx, p)
	return chunks, err
}

// ScheduleWithID is [Client.Schedule] that also returns the job's request_id,
// so a caller can surface it (e.g. an X-Mass-Job-Id header) before streaming
// and reconnect via [Client.GetResult] if the stream is dropped.
func (c *Client) ScheduleWithID(ctx context.Context, p ScheduleParams) (string, <-chan JobChunk, error) {
	jobID, err := c.SubmitOnly(ctx, p)
	if err != nil {
		return "", nil, err
	}
	out := make(chan JobChunk, 16)
	go c.streamWithResume(ctx, jobID, out)
	return jobID, out, nil
}

// Reattach re-opens the chunk stream for an already-submitted job by id,
// without creating a new one. A running job streams from where it left off;
// a finished job replays its terminal frame from MASS's durable store (within
// the result TTL). This is how a client that dropped the original connection
// resumes: same blocking stream, no re-run. Returns the live channel; callers
// distinguish "unknown/expired" via the terminal ChunkError frame.
func (c *Client) Reattach(ctx context.Context, jobID string) <-chan JobChunk {
	out := make(chan JobChunk, 16)
	go c.streamWithResume(ctx, jobID, out)
	return out
}

// SubmitOnly enqueues a job and returns its request_id without draining the
// result stream. It's the entry point for the async API: the caller holds the
// id and fetches the outcome later via [Client.GetResult]. The chunks are not
// streamed — MASS persists the terminal result in its durable store.
func (c *Client) SubmitOnly(ctx context.Context, p ScheduleParams) (string, error) {
	resp, err := c.c.Submit(ctx, &gatewaypb.SubmitRequest{
		ModelId:       p.ModelID,
		Payload:       p.Payload,
		Cost:          p.Cost,
		CostAxis:      p.CostAxis,
		Files:         p.Files,
		LoadHints:     p.LoadHints,
		BaseLoadBytes: p.BaseLoadBytes,
		PerSlotBytes:  p.PerSlotBytes,
		HeadroomPct:   p.HeadroomPct,
		Source:        p.Source,
		Priority:      p.Priority,
	})
	if err != nil {
		return "", ctxerr.With(fmt.Errorf("submit job: %w", err), map[string]any{"model_id": p.ModelID})
	}
	return resp.GetJobId(), nil
}

// Result is the durable outcome of a submitted job, fetched by request_id.
// Body is set when Status is ResultStatusDone; Err when ResultStatusError.
type Result struct {
	Status ResultStatus
	Body   []byte
	Err    string
}

// ResultStatus mirrors the job lifecycle in MASS's durable store.
type ResultStatus int

const (
	ResultPending ResultStatus = iota
	ResultProcessing
	ResultDone
	ResultError
)

// GetResult fetches a submitted job's durable result by request_id. Returns
// [ErrResultNotFound] when MASS has no result for the id (unknown or expired
// past the result TTL). A still-running job returns a Result with status
// ResultPending or ResultProcessing and an empty body.
func (c *Client) GetResult(ctx context.Context, requestID string) (*Result, error) {
	resp, err := c.c.GetResult(ctx, &gatewaypb.GetResultRequest{RequestId: requestID})
	if err != nil {
		if status.Code(err) == codes.NotFound {
			return nil, ErrResultNotFound
		}
		return nil, ctxerr.With(fmt.Errorf("get result: %w", err), map[string]any{"request_id": requestID})
	}
	return &Result{Status: resultStatusFromProto(resp.GetStatus()), Body: resp.GetBody(), Err: resp.GetError()}, nil
}

// CancelJob cancels a submitted job by request_id, whether it is pending or
// running. Returns [ErrResultNotFound] when no live job matches (already
// finished, never existed, or expired).
func (c *Client) CancelJob(ctx context.Context, requestID string) error {
	_, err := c.c.CancelJob(ctx, &gatewaypb.CancelJobRequest{RequestId: requestID})
	if err != nil {
		if status.Code(err) == codes.NotFound {
			return ErrResultNotFound
		}
		return ctxerr.With(fmt.Errorf("cancel job: %w", err), map[string]any{"request_id": requestID})
	}
	return nil
}

// ErrResultNotFound is returned when MASS has no live/stored result for a
// request_id — unknown, expired, or (for cancel) not cancellable.
var ErrResultNotFound = errors.New("result not found")

func resultStatusFromProto(s gatewaypb.ResultStatus) ResultStatus {
	switch s {
	case gatewaypb.ResultStatus_RESULT_STATUS_PROCESSING:
		return ResultProcessing
	case gatewaypb.ResultStatus_RESULT_STATUS_DONE:
		return ResultDone
	case gatewaypb.ResultStatus_RESULT_STATUS_ERROR:
		return ResultError
	default:
		return ResultPending
	}
}

// ErrStreamTruncated is the stream ending without a terminal frame. MASS always
// sends one, so a clean EOF before it means the stream was cut short — the job
// itself is still durable on the MASS side and can be resumed by seq.
var ErrStreamTruncated = errors.New("stream closed without terminal frame")

// streamWithResume opens StreamChunks for jobID, drains it, and retries
// once from the last observed seq when the stream breaks. A terminal
// frame (Completed / Error) or context cancellation ends the loop.
func (c *Client) streamWithResume(ctx context.Context, jobID string, out chan<- JobChunk) {
	defer close(out)

	resumeSeq := uint64(0)
	retried := false
	for {
		nextSeq, terminal, err := c.drainStreamChunks(ctx, jobID, resumeSeq, out)
		switch {
		case terminal:
			return
		case errors.Is(err, context.Canceled), errors.Is(ctx.Err(), context.Canceled):
			return
		case !retried:
			c.logger.Warn().Err(err).Str("job_id", jobID).Uint64("resume_seq", nextSeq).Msg("stream_chunks dropped; retrying once")
			retried = true
			resumeSeq = nextSeq
			continue
		default:
			out <- JobChunk{Type: ChunkError, ErrText: err.Error()}
			return
		}
	}
}

// drainStreamChunks opens one StreamChunks call and pumps frames into out.
// Returns (nextSeq, terminal, error) where nextSeq is the resume_seq the
// caller should pass on a retry, and terminal is true when a Completed
// or Error frame was observed (no retry needed). Every non-terminal return
// carries an error — a stream that stops without a terminal frame yields
// [ErrStreamTruncated] — so the caller never has to treat "no frames, no
// error" as a third outcome.
func (c *Client) drainStreamChunks(ctx context.Context, jobID string, resumeSeq uint64, out chan<- JobChunk) (uint64, bool, error) {
	stream, err := c.c.StreamChunks(ctx, &gatewaypb.StreamChunksRequest{
		JobId:     jobID,
		ResumeSeq: resumeSeq,
	})
	if err != nil {
		return resumeSeq, false, ctxerr.With(fmt.Errorf("opening stream_chunks: %w", err), map[string]any{"job_id": jobID, "resume_seq": resumeSeq})
	}

	nextSeq := resumeSeq
	for {
		frame, err := stream.Recv()
		if err != nil {
			if errors.Is(err, io.EOF) {
				return nextSeq, false, ErrStreamTruncated
			}
			return nextSeq, false, err
		}
		nextSeq = frame.GetSeq() + 1
		switch f := frame.GetFrame().(type) {
		case *gatewaypb.ScheduleChunk_Chunk:
			out <- JobChunk{Type: ChunkBody, Chunk: f.Chunk}
		case *gatewaypb.ScheduleChunk_Progress:
			out <- JobChunk{Type: ChunkProgress, ProgressPct: f.Progress.GetPct(), ProgressNote: f.Progress.GetNote()}
		case *gatewaypb.ScheduleChunk_Completed:
			out <- JobChunk{Type: ChunkCompleted, Final: f.Completed.GetFinalResponse(), WorkerID: f.Completed.GetWorkerId()}
			return nextSeq, true, nil
		case *gatewaypb.ScheduleChunk_Error:
			out <- JobChunk{Type: ChunkError, ErrText: f.Error.GetMessage(), WorkerID: f.Error.GetWorkerId()}
			return nextSeq, true, nil
		}
	}
}

// EvictModel asks MASS to drop a model from one (or all) workers.
func (c *Client) EvictModel(ctx context.Context, modelID, workerID string) (int32, error) {
	resp, err := c.c.EvictModel(ctx, &gatewaypb.EvictModelRequest{ModelId: modelID, WorkerId: workerID})
	if err != nil {
		return 0, ctxerr.With(fmt.Errorf("evict_model: %w", err), map[string]any{"model_id": modelID, "worker_id": workerID})
	}
	return resp.GetEvictedCount(), nil
}

// ListWorkers returns the gateway's view of online workers of its kind.
func (c *Client) ListWorkers(ctx context.Context) ([]*gatewaypb.WorkerSummary, error) {
	resp, err := c.c.ListWorkers(ctx, &gatewaypb.ListWorkersRequest{})
	if err != nil {
		return nil, ctxerr.With(fmt.Errorf("list_workers: %w", err), nil)
	}
	return resp.GetWorkers(), nil
}

// DownloadFiles asks MASS to enqueue a list of files into its
// download manager. Used by the gateway's install UI to ship the
// resolved file set after the operator picks a model in the
// registry picker. Returns the number of files MASS accepted into
// the queue.
func (c *Client) DownloadFiles(ctx context.Context, groupName string, files []*gatewaypb.DownloadFile) (int32, error) {
	resp, err := c.c.DownloadFiles(ctx, &gatewaypb.DownloadFilesRequest{
		Files:     files,
		GroupName: groupName,
	})
	if err != nil {
		return 0, ctxerr.With(fmt.Errorf("download_files: %w", err), map[string]any{"group_name": groupName, "file_count": len(files)})
	}
	return resp.GetQueued(), nil
}
