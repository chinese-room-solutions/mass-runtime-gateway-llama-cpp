package gateway

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/KernelPryanic/ctxerr"
	gatewaypb "github.com/chinese-room-solutions/mass-proto/gen/go/gateway"
	llamacppv1 "github.com/chinese-room-solutions/mass-runtime-llama-cpp/gen/go/llama_cpp/v1"
	"github.com/chinese-room-solutions/mass-runtime-llama-cpp/internal/sched"
	hf "github.com/chinese-room-solutions/mass-sdk/huggingface"
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
	schedConn *grpc.ClientConn // brokered connection to MASS's MassScheduler
	scheduler *sched.Client    // typed wrapper over schedConn
	cache     *metadataCache   // GGUF parse cache (memory + on-disk under dataDir)
	router    http.Handler     // built once Init has run; routes /mass.llama-cpp.* + /v1/*
	grpcSrv   *grpc.Server     // typed LlamaCppService; ServeHTTP-mounted for gRPC requests
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

	if err := os.MkdirAll(req.GetModelsDir(), 0o755); err != nil {
		return nil, ctxerr.With(fmt.Errorf("creating models dir: %w", err), map[string]any{"path": req.GetModelsDir()})
	}

	g.mu.Lock()
	g.dataDir = req.GetDataDir()
	g.modelsDir = req.GetModelsDir()
	g.schedConn = conn
	g.scheduler = sched.NewClient(conn, g.logger)
	g.cache = newMetadataCache(g.dataDir, g.logger)
	deps := routerDeps{
		params:      g.params,
		runtimeName: g.params.RuntimeName,
		modelsDir:   g.modelsDir,
		scheduler:   g.scheduler,
		cache:       g.cache,
		logger:      g.logger,
	}
	g.router = newRouter(deps)
	g.grpcSrv = grpc.NewServer()
	llamacppv1.RegisterLlamaCppServiceServer(g.grpcSrv, newGRPCServer(newHandlers(deps)))
	g.mu.Unlock()

	g.logger.Info().Str("data_dir", req.GetDataDir()).Str("models_dir", req.GetModelsDir()).Str("log_level", req.GetLogLevel()).Msg("gateway initialised")

	_ = ctx
	return &gatewaypb.InitResponse{
		RuntimeName: g.params.RuntimeName,
		Version:     g.params.Version,
		DisplayName: g.params.DisplayName,
		Description: "Runtime gateway for llama.cpp-family inference workers.",
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
	grpcSrv := g.grpcSrv
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

	// MASS strips its `/mass.<runtime_name>` prefix and forwards the rest
	// verbatim, so the path arrives as ".v1/Foo" (leading dot, no slash).
	// http.NewRequestWithContext parses that as a relative URL, which the
	// mux can't dispatch and falls through to its trailing-slash redirect.
	// Force a leading "/" so the mux sees an absolute path it can route on.
	reqPath := first.Path
	if reqPath == "" || reqPath[0] != '/' {
		reqPath = "/" + reqPath
	}
	httpReq, err := http.NewRequestWithContext(stream.Context(), first.Method, reqPath, body)
	if err != nil {
		return status.Errorf(codes.InvalidArgument, "constructing http request: %v", err)
	}
	for k, v := range first.GetHeaders() {
		httpReq.Header.Set(k, v)
	}

	rw := newStreamResponseWriter(stream)
	if isGRPCRequest(httpReq) && grpcSrv != nil {
		// gRPC requires HTTP/2; the reconstructed request defaults to 1.1.
		// Forging the proto fields satisfies grpc.Server.ServeHTTP's check
		// without us running an actual h2 connection — the body has already
		// been buffered, and our streamResponseWriter implements http.Flusher,
		// which is all the server-handler transport needs.
		httpReq.ProtoMajor = 2
		httpReq.ProtoMinor = 0
		httpReq.Proto = "HTTP/2.0"
		grpcSrv.ServeHTTP(rw, httpReq)
	} else {
		router.ServeHTTP(rw, httpReq)
	}
	return rw.Finish()
}

// isGRPCRequest reports whether the inbound HTTP request looks like a gRPC
// call (per the application/grpc* content-type convention). gRPC also
// constrains method=POST, but we leave that to the gRPC server's own
// validation so non-POST gRPC paths get a proper gRPC error rather than
// silently falling through to the HTTP router.
func isGRPCRequest(r *http.Request) bool {
	ct := r.Header.Get("Content-Type")
	return strings.HasPrefix(ct, "application/grpc")
}

// ListGroups is the gateway's authoritative, fully-grouped catalogue.
// It walks <modelsDir>/gguf/, parses every recognised file (caching per
// (path, mtime, size) so repeat calls are cheap), and returns one
// [gatewaypb.Group] per operator-typed group name; each Group holds
// its child [gatewaypb.Model] files (different quants, mmproj
// companions, etc.).
//
// Grouping is a runtime concern — MASS receives the groups as-is and
// renders them.
func (g *Gateway) ListGroups(ctx context.Context, _ *gatewaypb.GatewayListGroupsRequest) (*gatewaypb.GatewayListGroupsResponse, error) {
	_ = ctx
	g.mu.RLock()
	dir := g.modelsDir
	cache := g.cache
	g.mu.RUnlock()
	if dir == "" || cache == nil {
		return nil, status.Error(codes.FailedPrecondition, "list_groups: models_dir not set (Init not run?)")
	}

	root := formatRoot(dir)
	infos, err := cache.walkAndParseModels(root)
	if err != nil {
		return nil, ctxerr.With(fmt.Errorf("walking models dir %s: %w", root, err), map[string]any{"dir": root})
	}
	return &gatewaypb.GatewayListGroupsResponse{Groups: groupModels(infos)}, nil
}

// PlanModelFiles maps an install request to the concrete file set
// MASS must fetch. For HuggingFace + GGUF: the requested file plus,
// when the repo also ships an mmproj projector, that companion file.
// The gateway hits the HF tree API once to size the files and detect
// the companion; MASS does the actual downloads.
//
// group_name is the operator-typed identifier (required) — every
// file produced by this plan gets that name stamped into the
// catalogue, so re-installs of related files (different quants, the
// projector, etc.) under the same name cluster into one Group.
//
// The mmproj-by-filename companion match is a pre-download bundling
// heuristic — the only signal available before bytes land. The
// projector's own GGUF header (architecture=clip) confirms its role
// at walk time.
func (g *Gateway) PlanModelFiles(ctx context.Context, req *gatewaypb.PlanModelFilesRequest) (*gatewaypb.PlanModelFilesResponse, error) {
	source := req.GetSource()
	if source != "huggingface" {
		return nil, status.Error(codes.InvalidArgument, "plan_model_files: only source=\"huggingface\" is supported by llama-cpp")
	}
	repoID := strings.TrimSpace(req.GetRepoId())
	filename := strings.TrimSpace(req.GetFilename())
	groupName := strings.TrimSpace(req.GetGroupName())
	if repoID == "" || filename == "" {
		return nil, status.Error(codes.InvalidArgument, "plan_model_files: repo_id and filename are required")
	}
	if groupName == "" {
		return nil, status.Error(codes.InvalidArgument, "plan_model_files: group_name is required (it becomes the group's display label and clusters files under one Group)")
	}

	files, err := hf.ListFiles(ctx, repoID, []string{".gguf"})
	if err != nil {
		return nil, ctxerr.With(fmt.Errorf("listing HF files: %w", err), map[string]any{"repo_id": repoID})
	}

	primaryIsProjector := looksLikeMmprojFilename(filename)

	var primarySize int64 = -1
	var mmprojName string
	var mmprojSize int64 = -1
	for _, f := range files {
		if f.Filename == filename {
			primarySize = f.SizeBytes
			continue
		}
		// Only auto-attach an mmproj companion when the user picked a
		// non-projector primary. Picking the projector itself is an
		// explicit single-file install and shouldn't pull siblings.
		if primaryIsProjector || mmprojName != "" {
			continue
		}
		if looksLikeMmprojFilename(f.Filename) {
			mmprojName = f.Filename
			mmprojSize = f.SizeBytes
		}
	}
	if primarySize < 0 {
		return nil, ctxerr.With(fmt.Errorf("file %q not found in repo %q", filename, repoID), map[string]any{"repo_id": repoID, "filename": filename})
	}

	primaryDest := planDestPath(g, groupName, filename)
	if err := assertDestNotTaken(g.modelsDir, primaryDest); err != nil {
		return nil, err
	}
	g.cache.reserveEntry(filepath.Join(g.modelsDir, primaryDest), groupName)

	out := []*gatewaypb.DownloadFile{
		{
			Url:        hfFileURL(repoID, filename),
			RelPath:    primaryDest,
			SizeBytes:  primarySize,
			GroupLabel: groupName,
		},
	}
	if mmprojName != "" {
		mmprojDest := planDestPath(g, groupName, mmprojName)
		if err := assertDestNotTaken(g.modelsDir, mmprojDest); err != nil {
			return nil, err
		}
		g.cache.reserveEntry(filepath.Join(g.modelsDir, mmprojDest), groupName)
		out = append(out, &gatewaypb.DownloadFile{
			Url:        hfFileURL(repoID, mmprojName),
			RelPath:    mmprojDest,
			SizeBytes:  mmprojSize,
			GroupLabel: groupName,
		})
	}
	return &gatewaypb.PlanModelFilesResponse{Files: out}, nil
}

// PlanLocalImport plans a Browse Local install. The operator picks
// one or more files in the dialog and types one group name; the
// gateway returns the full list of files MASS should copy under
// models_dir. Every file gets the same name stamped into the
// catalogue so they cluster as one Group.
//
// Returns codes.InvalidArgument when src_path or group_name is empty.
// Returns codes.AlreadyExists when the destination filename collides.
func (g *Gateway) PlanLocalImport(_ context.Context, req *gatewaypb.PlanLocalImportRequest) (*gatewaypb.PlanLocalImportResponse, error) {
	src := strings.TrimSpace(req.GetSrcPath())
	if src == "" {
		return nil, status.Error(codes.InvalidArgument, "plan_local_import: src_path is required")
	}
	groupName := strings.TrimSpace(req.GetGroupName())
	if groupName == "" {
		return nil, status.Error(codes.InvalidArgument, "plan_local_import: group_name is required (it becomes the group's display label and clusters files under one Group)")
	}
	if !strings.EqualFold(extOf(src), "gguf") {
		return nil, status.Errorf(codes.NotFound, "plan_local_import: %s is not a GGUF file", filepath.Base(src))
	}

	dest := planDestPath(g, groupName, src)
	if err := assertDestNotTaken(g.modelsDir, dest); err != nil {
		return nil, err
	}
	g.cache.reserveEntry(filepath.Join(g.modelsDir, dest), groupName)

	size := int64(-1)
	if st, err := os.Stat(src); err == nil {
		size = st.Size()
	}
	return &gatewaypb.PlanLocalImportResponse{
		Files: []*gatewaypb.DownloadFile{{
			Url:        "file://" + src,
			RelPath:    dest,
			SizeBytes:  size,
			GroupLabel: groupName,
		}},
	}, nil
}

// RenameGroup atomically rewrites every catalogue entry whose Name
// matches id to use new_name. The id MASS holds is the slug of the
// current Name (see groupModels); we resolve it back to the source
// Name by scanning the catalogue. Returns NotFound when no entry
// matches the slug.
func (g *Gateway) RenameGroup(_ context.Context, req *gatewaypb.RenameGroupRequest) (*gatewaypb.RenameGroupResponse, error) {
	id := strings.TrimSpace(req.GetId())
	newName := strings.TrimSpace(req.GetNewName())
	if id == "" || newName == "" {
		return nil, status.Error(codes.InvalidArgument, "rename_group: id and new_name are required")
	}
	old := g.cache.nameForSlug(id)
	if old == "" {
		return nil, status.Errorf(codes.NotFound, "rename_group: no group matches id %q", id)
	}
	if old == newName {
		return &gatewaypb.RenameGroupResponse{}, nil
	}
	n, err := g.cache.renameGroup(g.modelsDir, old, newName)
	if err != nil {
		if errors.Is(err, errEntryReserved) {
			return nil, status.Errorf(codes.FailedPrecondition, "rename_group: %v", err)
		}
		return nil, ctxerr.With(fmt.Errorf("renaming group %q → %q: %w", old, newName, err), map[string]any{"old": old, "new": newName})
	}
	if n == 0 {
		return nil, status.Errorf(codes.NotFound, "rename_group: no entries renamed for %q", old)
	}
	g.cache.saveToDisk()
	g.logger.Info().Str("old", old).Str("new", newName).Int("entries", n).Msg("renamed group")
	return &gatewaypb.RenameGroupResponse{}, nil
}

// sanitiseFilename strips path-unsafe characters from a source basename
// so it can land on disk verbatim. The OS-dangerous set (/, \, :, *,
// etc.) plus control chars get replaced with '-'; surrounding whitespace
// and dots are trimmed.
func sanitiseFilename(name string) string {
	var b strings.Builder
	for _, r := range strings.TrimSpace(name) {
		switch {
		case r == '/' || r == '\\' || r == ':' || r == '*' || r == '?' || r == '"' || r == '<' || r == '>' || r == '|':
			b.WriteByte('-')
		case r < 0x20:
			// drop control chars
		default:
			b.WriteRune(r)
		}
	}
	return strings.Trim(b.String(), " .-")
}

// planDestPath returns the under-modelsDir relPath where a source
// file should land. Layout: "<formatDir>/<group-slug>/<sanitised
// source basename>". The per-group subdirectory mirrors the Models
// tab grouping on disk so the catalogue and the filesystem agree on
// "this file belongs to that group" without operator inspection.
func planDestPath(_ *Gateway, groupName, srcName string) string {
	return filepath.ToSlash(filepath.Join(formatDir, modelSlug(groupName), sanitiseFilename(filepath.Base(srcName))))
}

// assertDestNotTaken returns codes.AlreadyExists when a file already
// occupies the target relPath under modelsDir. Caller (PlanModelFiles /
// PlanLocalImport) surfaces the error to the operator instead of
// silently overwriting.
func assertDestNotTaken(modelsDir, relPath string) error {
	if _, err := os.Stat(filepath.Join(modelsDir, relPath)); err == nil {
		return status.Errorf(codes.AlreadyExists, "destination already taken: %s", relPath)
	}
	return nil
}

// hfFileURL is the canonical resolve URL for a file in a HuggingFace repo.
func hfFileURL(repoID, filename string) string {
	return "https://huggingface.co/" + repoID + "/resolve/main/" + url.PathEscape(filename)
}

// looksLikeMmprojFilename is the pre-download heuristic for spotting
// a mmproj projector by community-convention filename. Used only by
// PlanModelFiles for HF companion bundling and by FilterFilenames to
// hide projectors from the picker — both run before any header is
// available. Identity (catalogue Name) is operator-typed; this is
// purely a fetch-time bundling hint.
func looksLikeMmprojFilename(name string) bool {
	lower := strings.ToLower(name)
	return strings.Contains(lower, "mmproj") && strings.HasSuffix(lower, ".gguf")
}

// capabilitiesFromHeader derives the gateway's Capabilities flags
// from header signals only. Vision is determined at walk time by
// scanning for a sibling projector under the same Name (the primary's
// own header doesn't say "I have vision"). Audio: not currently
// inferable from llama.cpp GGUFs; left false. Thinking: chat template
// substring detection — gated on the file actually being a chat
// model, since some embedding models (Qwen3-Embedding) also ship a
// chat template containing think tokens but don't emit reasoning.
func capabilitiesFromHeader(kv map[string]string) *gatewaypb.Capabilities {
	thinking := strings.EqualFold(kv["thinking"], "true") && modelTypeFromHeader(kv) == "chat"
	return &gatewaypb.Capabilities{
		Thinking: thinking,
	}
}

// modelTypeFromHeader distinguishes chat from embedding. Projectors
// (clip architecture) return "" — not a standalone model_type.
//
// Signal priority: <arch>.pooling_type wins because some embedding
// models (Qwen3-Embedding, etc.) also ship a chat template, which
// makes template-presence alone misclassify them as chat. Pooling
// type is set by embedding models and unset by chat models.
//
// Fallback to chat-template presence for older / loosely-tagged
// GGUFs that don't declare pooling_type.
func modelTypeFromHeader(kv map[string]string) string {
	if strings.EqualFold(kv["architecture"], "clip") {
		return ""
	}
	if strings.EqualFold(kv["pooling_type_present"], "true") {
		return "embedding"
	}
	if strings.EqualFold(kv["chat_template_present"], "true") {
		return "chat"
	}
	return "embedding"
}

// parsedModel is the gateway's internal, fully-decorated view of one
// model file. Not part of any RPC — gateway code projects this into
// the proto Model (for MASS) or directly into the runtime's app-facing
// typed responses. Name is the operator-typed identity; everything
// else is header-derived display data.
type parsedModel struct {
	ID           string // store-relative id (basename of disk filename)
	AbsolutePath string
	SizeBytes    int64
	Name         string
	VariantLabel string
	Companion    string // "mmproj" for vision projectors; "" otherwise
	ModelType    string // "chat" / "embedding" / ""
	Capabilities *gatewaypb.Capabilities
	Properties   map[string]string // metadata for the detail panel
}

// propertiesFromKV strips header keys already surfaced as first-class
// fields and returns the rest for the Models detail panel.
func propertiesFromKV(kv map[string]string) map[string]string {
	if len(kv) == 0 {
		return nil
	}
	skip := map[string]struct{}{
		"thinking":              {},
		"quant":                 {},
		"organization":          {},
		"basename":              {},
		"size_label":            {},
		"finetune":              {},
		"version":               {},
		"quantized_by":          {},
		"chat_template_present": {},
		"pooling_type_present":  {},
	}
	out := make(map[string]string, len(kv))
	for k, v := range kv {
		if _, drop := skip[k]; drop {
			continue
		}
		out[k] = v
	}
	return out
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
	headers := flattenResponseHeaders(w.header)
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

// Finish emits the terminal end-of-stream frame, attaching any trailers the
// handler wrote (either via `Trailer:` pre-declaration or via the
// http.TrailerPrefix late-write convention). Called by HandleRequest after
// the handler returns.
//
// Trailers matter for gRPC: the in-process gRPC server emits grpc-status /
// grpc-message as HTTP/2 trailers; without them the client sees a hung RPC.
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
	trailers := extractTrailers(w.header)
	if err := w.stream.Send(&gatewaypb.HTTPResponseChunk{
		EndOfStream: true,
		Trailers:    trailers,
	}); err != nil {
		return ctxerr.With(fmt.Errorf("sending EOS: %w", err), nil)
	}
	return nil
}

// extractTrailers harvests trailer entries from an http.Header per net/http's
// two trailer conventions:
//
//   - Pre-declared trailers: keys listed in the comma-separated "Trailer"
//     header before WriteHeader. Their final values live under those keys
//     directly in w.Header() after the handler returns.
//   - Late trailers: values written under the http.TrailerPrefix
//     ("Trailer:") magic prefix. The prefix is stripped here.
//
// Returns nil when no trailers exist (so plain HTTP responses get an empty
// frame field, matching the proto's "empty for plain HTTP" doc).
func extractTrailers(h http.Header) map[string]string {
	var out map[string]string
	put := func(key string, vs []string) {
		if len(vs) == 0 {
			return
		}
		if out == nil {
			out = make(map[string]string)
		}
		out[key] = strings.Join(vs, ",")
	}
	if announced := h.Get("Trailer"); announced != "" {
		for _, k := range strings.Split(announced, ",") {
			k = http.CanonicalHeaderKey(strings.TrimSpace(k))
			if k == "" {
				continue
			}
			put(k, h.Values(k))
		}
	}
	for k, vs := range h {
		if !strings.HasPrefix(k, http.TrailerPrefix) {
			continue
		}
		put(strings.TrimPrefix(k, http.TrailerPrefix), vs)
	}
	return out
}

// Compile-time assertions.
var (
	_ http.ResponseWriter = (*streamResponseWriter)(nil)
	_ http.Flusher        = (*streamResponseWriter)(nil)
)

// flattenResponseHeaders flattens the response header map for the initial
// HTTPResponseChunk. Trailer entries are excluded — late-bound values written
// under http.TrailerPrefix and any names listed in the announced "Trailer"
// header are sent separately on the EndOfStream chunk.
func flattenResponseHeaders(in http.Header) map[string]string {
	announced := make(map[string]struct{})
	if v := in.Get("Trailer"); v != "" {
		for _, k := range strings.Split(v, ",") {
			k = http.CanonicalHeaderKey(strings.TrimSpace(k))
			if k != "" {
				announced[k] = struct{}{}
			}
		}
	}
	out := make(map[string]string, len(in))
	for k, vs := range in {
		if strings.HasPrefix(k, http.TrailerPrefix) {
			continue
		}
		if _, isTrailer := announced[k]; isTrailer {
			continue
		}
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
