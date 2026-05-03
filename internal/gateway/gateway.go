package gateway

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/KernelPryanic/ctxerr"
	gatewaypb "github.com/chinese-room-solutions/mass-proto/gen/go/gateway"
	"github.com/chinese-room-solutions/mass-runtime-llama-cpp/internal/gguf"
	"github.com/chinese-room-solutions/mass-runtime-llama-cpp/internal/sched"
	"github.com/hashicorp/go-plugin"
	"github.com/rs/zerolog"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// Gateway implements gatewaypb.RuntimeGatewayServer for llama-cpp.
type Gateway struct {
	gatewaypb.UnimplementedRuntimeGatewayServer

	params PluginParams
	broker *plugin.GRPCBroker

	logger zerolog.Logger

	mu        sync.RWMutex
	dataDir   string
	modelsDir string
	schedConn *grpc.ClientConn   // brokered connection to MASS's MassScheduler
	scheduler *sched.Client      // typed wrapper over schedConn
	router    http.Handler       // built once Init has run; routes /mass.llama-cpp.* + /v1/*
}

// newGateway constructs a Gateway. The broker is needed in Init to dial back
// to MASS; the actual dial is deferred until Init supplies the broker ID.
func newGateway(params PluginParams, broker *plugin.GRPCBroker) *Gateway {
	return &Gateway{
		params: params,
		broker: broker,
		logger: params.Logger.With().Str("component", "gateway").Logger(),
	}
}

// Init runs once on plugin start. MASS hands us the data + models dirs, our
// log level, and the broker ID for the MassScheduler callback service.
func (g *Gateway) Init(ctx context.Context, req *gatewaypb.InitRequest) (*gatewaypb.InitResponse, error) {
	if req.GetMassSchedulerBrokerId() == 0 {
		return nil, status.Error(codes.InvalidArgument, "init: mass_scheduler_broker_id is required")
	}

	if level, err := zerolog.ParseLevel(req.GetLogLevel()); err == nil && req.GetLogLevel() != "" {
		zerolog.SetGlobalLevel(level)
	}

	conn, err := g.broker.Dial(req.GetMassSchedulerBrokerId())
	if err != nil {
		return nil, ctxerr.With(fmt.Errorf("dialing mass scheduler: %w", err), map[string]any{"broker_id": req.GetMassSchedulerBrokerId()})
	}

	g.mu.Lock()
	g.dataDir = req.GetDataDir()
	g.modelsDir = req.GetModelsDir()
	g.schedConn = conn
	g.scheduler = sched.NewClient(conn, g.logger)
	g.router = newRouter(routerDeps{
		params:    g.params,
		modelsDir: g.modelsDir,
		scheduler: g.scheduler,
		logger:    g.logger,
	})
	g.mu.Unlock()

	g.logger.Info().Str("data_dir", req.GetDataDir()).Str("models_dir", req.GetModelsDir()).Str("log_level", req.GetLogLevel()).Msg("gateway initialised")

	_ = ctx
	return &gatewaypb.InitResponse{
		RuntimeKind:      g.params.RuntimeKind,
		Version:          g.params.Version,
		DisplayName:      g.params.DisplayName,
		Description:      "Runtime gateway for llama.cpp-family inference workers.",
		SupportedFormats: []string{"gguf"},
	}, nil
}

// HandleRequest is the inbound HTTP-over-gRPC entrypoint. We reassemble the
// streamed request frames into a normal http.Request, route it via our
// http.ServeMux, and stream the response back as HTTPResponseChunk frames.
//
// The streaming response writer flushes whenever the caller writes (so SSE
// chunks travel through immediately), which is what the OpenAI-compat
// streaming chat path needs.
func (g *Gateway) HandleRequest(stream gatewaypb.RuntimeGateway_HandleRequestServer) error {
	g.mu.RLock()
	router := g.router
	g.mu.RUnlock()
	if router == nil {
		return status.Error(codes.FailedPrecondition, "gateway: HandleRequest called before Init")
	}

	first, err := stream.Recv()
	if err != nil {
		return ctxerr.With(fmt.Errorf("receiving first request chunk: %w", err), nil)
	}
	if first.GetMethod() == "" || first.GetPath() == "" {
		return status.Error(codes.InvalidArgument, "first chunk must carry method + path")
	}

	body, bodyErr := assembleRequestBody(stream, first)
	if bodyErr != nil {
		return bodyErr
	}

	httpReq, err := http.NewRequestWithContext(stream.Context(), first.Method, first.Path, body)
	if err != nil {
		return status.Errorf(codes.InvalidArgument, "constructing http request: %v", err)
	}
	for k, v := range first.GetHeaders() {
		httpReq.Header.Set(k, v)
	}

	rw := newStreamResponseWriter(stream)
	router.ServeHTTP(rw, httpReq)
	return rw.Finish()
}

// ParseModel reads metadata from a model file. Only GGUF is recognised today;
// any other extension returns Recognized=false (not an error).
//
// Capability flags surfaced into MetadataKv:
//   - "thinking" = "true" when the chat template contains reasoning markers.
//   - "has_vision" = "true" for chat models with a sibling mmproj-*.gguf in
//     the same directory.
//   - "companion" = "mmproj" when the file itself is a vision projector
//     (architecture=clip). MASS treats companions as plumbing — they're
//     hidden from top-level grouping and shown alongside the base model
//     they pair with.
func (g *Gateway) ParseModel(ctx context.Context, req *gatewaypb.ParseModelRequest) (*gatewaypb.ParseModelResponse, error) {
	_ = ctx
	path := req.GetPath()
	if !strings.EqualFold(extOf(path), "gguf") {
		return &gatewaypb.ParseModelResponse{Recognized: false}, nil
	}
	meta, err := gguf.ReadMeta(path)
	if err != nil {
		return nil, ctxerr.With(fmt.Errorf("reading gguf: %w", err), map[string]any{"path": path})
	}
	kv := meta.Summary()
	isMmproj := strings.EqualFold(kv["architecture"], "clip")
	if isMmproj {
		kv["companion"] = "mmproj"
	} else if hasMmprojSibling(path) {
		kv["has_vision"] = "true"
	}
	return &gatewaypb.ParseModelResponse{
		Recognized: true,
		Format:     "gguf",
		MetadataKv: kv,
	}, nil
}

// hasMmprojSibling returns true when path's parent directory contains a
// file matching "mmproj*.gguf" (case-insensitive). Vision-capable chat
// models typically ship a separate mmproj projector file alongside the
// main weights.
func hasMmprojSibling(path string) bool {
	dir := filepath.Dir(path)
	entries, err := os.ReadDir(dir)
	if err != nil {
		return false
	}
	self := filepath.Base(path)
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		name := e.Name()
		if name == self {
			continue
		}
		low := strings.ToLower(name)
		if strings.HasPrefix(low, "mmproj") && strings.HasSuffix(low, ".gguf") {
			return true
		}
	}
	return false
}

// OnWorkerLoaded is currently a no-op. We don't pre-warm anything per-worker;
// loads happen on demand at the first request that needs the model.
func (g *Gateway) OnWorkerLoaded(_ context.Context, req *gatewaypb.OnWorkerLoadedRequest) (*gatewaypb.OnWorkerLoadedResponse, error) {
	g.logger.Debug().Str("worker_id", req.GetWorkerId()).Msg("worker connected")
	return &gatewaypb.OnWorkerLoadedResponse{}, nil
}

// OnWorkerLost is currently a no-op. The MASS-side scheduler index is the
// authority on loaded models; we let it diverge passively.
func (g *Gateway) OnWorkerLost(_ context.Context, req *gatewaypb.OnWorkerLostRequest) (*gatewaypb.OnWorkerLostResponse, error) {
	g.logger.Debug().Str("worker_id", req.GetWorkerId()).Msg("worker disconnected")
	return &gatewaypb.OnWorkerLostResponse{}, nil
}

// ----- Helpers -----

// assembleRequestBody collects the body bytes from the streamed first frame
// + subsequent frames into a single io.ReadCloser. We buffer the body fully
// before invoking the HTTP handler — chat / embed / tokenize requests are
// small (a few KB usually). If we ever need to stream uploads we'll switch
// to a goroutine-fed pipe; not worth the complexity today.
func assembleRequestBody(stream gatewaypb.RuntimeGateway_HandleRequestServer, first *gatewaypb.HTTPRequestChunk) (io.ReadCloser, error) {
	var buf bytes.Buffer
	if len(first.GetBody()) > 0 {
		buf.Write(first.GetBody())
	}
	if first.GetEndOfStream() {
		return io.NopCloser(&buf), nil
	}
	for {
		chunk, err := stream.Recv()
		if err != nil {
			if errors.Is(err, io.EOF) {
				return io.NopCloser(&buf), nil
			}
			return nil, ctxerr.With(fmt.Errorf("receiving request body chunk: %w", err), nil)
		}
		if len(chunk.GetBody()) > 0 {
			buf.Write(chunk.GetBody())
		}
		if chunk.GetEndOfStream() {
			return io.NopCloser(&buf), nil
		}
	}
}

// streamResponseWriter is an http.ResponseWriter that streams writes back as
// HTTPResponseChunk frames. The first Write (or first WriteHeader) flushes
// the status + headers. Subsequent writes are body frames; close emits the
// terminal end-of-stream marker.
type streamResponseWriter struct {
	stream      gatewaypb.RuntimeGateway_HandleRequestServer
	header      http.Header
	status      int
	wroteHeader bool
	finished    bool
	finishErr   error
}

func newStreamResponseWriter(stream gatewaypb.RuntimeGateway_HandleRequestServer) *streamResponseWriter {
	return &streamResponseWriter{stream: stream, header: http.Header{}}
}

func (w *streamResponseWriter) Header() http.Header { return w.header }

func (w *streamResponseWriter) WriteHeader(status int) {
	if w.wroteHeader {
		return
	}
	w.status = status
	w.wroteHeader = true
	headers := flattenHeaders(w.header)
	if err := w.stream.Send(&gatewaypb.HTTPResponseChunk{
		Status:  int32(status),
		Headers: headers,
	}); err != nil {
		w.finishErr = err
	}
}

func (w *streamResponseWriter) Write(p []byte) (int, error) {
	if !w.wroteHeader {
		w.WriteHeader(http.StatusOK)
	}
	if w.finishErr != nil {
		return 0, w.finishErr
	}
	if len(p) == 0 {
		return 0, nil
	}
	if err := w.stream.Send(&gatewaypb.HTTPResponseChunk{Body: p}); err != nil {
		w.finishErr = err
		return 0, err
	}
	return len(p), nil
}

// Flush is called by handlers using SSE/streaming responses. We always send
// every Write immediately, so this is a no-op besides confirming we
// implement http.Flusher.
func (w *streamResponseWriter) Flush() {}

// Finish emits the terminal end-of-stream frame. Called by HandleRequest
// after the handler returns.
func (w *streamResponseWriter) Finish() error {
	if w.finished {
		return w.finishErr
	}
	w.finished = true
	if !w.wroteHeader {
		w.WriteHeader(http.StatusOK)
	}
	if w.finishErr != nil {
		return w.finishErr
	}
	if err := w.stream.Send(&gatewaypb.HTTPResponseChunk{EndOfStream: true}); err != nil {
		return ctxerr.With(fmt.Errorf("sending EOS: %w", err), nil)
	}
	return nil
}

// Compile-time assertions.
var (
	_ http.ResponseWriter = (*streamResponseWriter)(nil)
	_ http.Flusher        = (*streamResponseWriter)(nil)
)

func flattenHeaders(in http.Header) map[string]string {
	out := make(map[string]string, len(in))
	for k, vs := range in {
		out[k] = strings.Join(vs, ",")
	}
	return out
}

func extOf(path string) string {
	i := strings.LastIndexByte(path, '.')
	if i < 0 || i == len(path)-1 {
		return ""
	}
	return path[i+1:]
}
