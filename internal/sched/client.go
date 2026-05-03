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
	Type        ChunkType
	Chunk       []byte
	Final       []byte
	ProgressPct float32
	ProgressNote string
	ErrText     string
	WorkerID    string
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
// to gatewaypb.ScheduleRequest is mechanical. Callers must call
// EnsureModelLoaded before Schedule when the model isn't already resident
// on a worker — Schedule itself does not auto-load.
type ScheduleParams struct {
	ModelID         string
	Payload         []byte
	Weight          int32
	AffinityWorkers []string
}

// Schedule submits a job and returns a channel of streamed chunks. The
// channel closes after a Completed or Error frame (or on context cancel).
func (c *Client) Schedule(ctx context.Context, p ScheduleParams) (<-chan JobChunk, error) {
	stream, err := c.c.Schedule(ctx, &gatewaypb.ScheduleRequest{
		ModelId:         p.ModelID,
		Payload:         p.Payload,
		Weight:          p.Weight,
		AffinityWorkers: p.AffinityWorkers,
	})
	if err != nil {
		return nil, ctxerr.With(fmt.Errorf("opening schedule stream: %w", err), map[string]any{"model_id": p.ModelID})
	}

	out := make(chan JobChunk, 16)
	go func() {
		defer close(out)
		for {
			frame, err := stream.Recv()
			if err != nil {
				if errors.Is(err, io.EOF) || errors.Is(ctx.Err(), context.Canceled) {
					return
				}
				out <- JobChunk{Type: ChunkError, ErrText: err.Error()}
				return
			}
			switch f := frame.GetFrame().(type) {
			case *gatewaypb.ScheduleChunk_Chunk:
				out <- JobChunk{Type: ChunkBody, Chunk: f.Chunk}
			case *gatewaypb.ScheduleChunk_Progress:
				out <- JobChunk{Type: ChunkProgress, ProgressPct: f.Progress.GetPct(), ProgressNote: f.Progress.GetNote()}
			case *gatewaypb.ScheduleChunk_Completed:
				out <- JobChunk{Type: ChunkCompleted, Final: f.Completed.GetFinalResponse(), WorkerID: f.Completed.GetWorkerId()}
				return
			case *gatewaypb.ScheduleChunk_Error:
				out <- JobChunk{Type: ChunkError, ErrText: f.Error.GetMessage(), WorkerID: f.Error.GetWorkerId()}
				return
			}
		}
	}()
	return out, nil
}

// LoadedInstance mirrors gatewaypb.LoadedInstance with our internal naming.
type LoadedInstance struct {
	WorkerID string
	PoolSize int32
}

// EnsureModelLoadedParams is the input set for EnsureModelLoaded.
type EnsureModelLoadedParams struct {
	ModelID   string
	Files     []*workerpb.ModelFile
	LoadHints []byte
	Preferred []string
	Source    string // who triggered the load (e.g. "app: playground", "direct"); MASS surfaces in Scheduler tab
}

// EnsureModelLoaded asks MASS to make sure the model is loaded somewhere.
// Returns the existing or new instances.
func (c *Client) EnsureModelLoaded(ctx context.Context, p EnsureModelLoadedParams) ([]LoadedInstance, error) {
	resp, err := c.c.EnsureModelLoaded(ctx, &gatewaypb.EnsureModelLoadedRequest{
		ModelId:          p.ModelID,
		Files:            p.Files,
		LoadHints:        p.LoadHints,
		PreferredWorkers: p.Preferred,
		Source:           p.Source,
	})
	if err != nil {
		return nil, ctxerr.With(fmt.Errorf("ensure_model_loaded: %w", err), map[string]any{"model_id": p.ModelID})
	}
	out := make([]LoadedInstance, len(resp.GetInstances()))
	for i, inst := range resp.GetInstances() {
		out[i] = LoadedInstance{
			WorkerID: inst.GetWorkerId(),
			PoolSize: inst.GetPoolSize(),
		}
	}
	return out, nil
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

