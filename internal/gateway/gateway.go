package gateway

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/KernelPryanic/ctxerr"
	gatewaypb "github.com/chinese-room-solutions/mass-proto/gen/go/gateway"
	llamacppv1 "github.com/chinese-room-solutions/mass-runtime-gateway-llama-cpp/gen/go/llama_cpp/v1"
	"github.com/chinese-room-solutions/mass-runtime-gateway-llama-cpp/internal/sched"
	"github.com/chinese-room-solutions/mass-sdk/gatewayhttp"
	hf "github.com/chinese-room-solutions/mass-sdk/huggingface"
	"github.com/chinese-room-solutions/mass-sdk/uikit"
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
	// Refuse to start when MASS speaks a wire version the gateway
	// wasn't built against. Gateway and MASS pin to the same constant
	// today; mismatches mean one side is out of date.
	if req.GetMassGatewayApiVersion() != gatewaypb.GatewayAPIVersion {
		return nil, status.Errorf(codes.FailedPrecondition,
			"gateway api version mismatch: MASS speaks %d, gateway built against %d",
			req.GetMassGatewayApiVersion(), gatewaypb.GatewayAPIVersion)
	}

	if level, err := zerolog.ParseLevel(req.GetLogLevel()); err == nil && req.GetLogLevel() != "" {
		zerolog.SetGlobalLevel(level)
	}

	// Load pluggable themes from the shared themes dir so the /install page
	// (rendered inside a MASS iframe that passes ?theme=<name>) can resolve
	// the same theme MASS selected. A bad theme file must not stop startup.
	if err := uikit.LoadThemes(); err != nil {
		g.logger.Warn().Err(err).Msg("loading pluggable uikit themes")
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
	g.cache = newMetadataCache(g.dataDir, g.modelsDir, g.logger)
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
		RuntimeName:       g.params.RuntimeName,
		Version:           g.params.Version,
		DisplayName:       g.params.DisplayName,
		Description:       g.params.Description,
		GatewayApiVersion: gatewaypb.GatewayAPIVersion,
		// Every llama-cpp worker is required to bench Q4_K matvec. MASS
		// uses this axis as the fallback when a Submit names an axis the
		// worker hasn't measured.
		DefaultCostAxis: q4kMatvecAxis,
	}, nil
}

// HandleRequest tunnels MASS's framed HTTP request into our router (or
// our typed gRPC server, when the content-type signals gRPC). The
// SDK's gatewayhttp.Serve owns the framing, body buffering, and
// streaming response — including HTTP/2 trailer round-tripping for
// gRPC.
func (g *Gateway) HandleRequest(stream gatewaypb.RuntimeGateway_HandleRequestServer) error {
	g.mu.RLock()
	router := g.router
	grpcSrv := g.grpcSrv
	g.mu.RUnlock()
	if router == nil {
		return status.Error(codes.FailedPrecondition, "gateway: HandleRequest called before Init")
	}
	return gatewayhttp.Serve(stream, router, grpcSrv)
}

// snapshot returns the Init-populated state every model RPC depends
// on. ok=false until Init has run — callers must translate that to
// FailedPrecondition instead of dereferencing a nil cache / racing an
// in-flight Init.
func (g *Gateway) snapshot() (modelsDir string, cache *metadataCache, ok bool) {
	g.mu.RLock()
	defer g.mu.RUnlock()
	return g.modelsDir, g.cache, g.modelsDir != "" && g.cache != nil
}

// errNotInitialised is the FailedPrecondition every model RPC returns
// when called before Init has populated the gateway's state.
func errNotInitialised(rpc string) error {
	return status.Errorf(codes.FailedPrecondition, "%s: gateway not initialised (Init not run)", rpc)
}

// ListGroups returns the gateway's catalogue: walks <modelsDir>/gguf/,
// parses recognised files (cache key: path, mtime, size), and returns
// one [gatewaypb.Group] per operator-typed group name. Grouping is
// runtime-private — MASS just renders.
func (g *Gateway) ListGroups(ctx context.Context, _ *gatewaypb.GatewayListGroupsRequest) (*gatewaypb.GatewayListGroupsResponse, error) {
	_ = ctx
	dir, cache, ok := g.snapshot()
	if !ok {
		return nil, errNotInitialised("list_groups")
	}

	root := formatRoot(dir)
	infos, err := cache.walkAndParseModels(root)
	if err != nil {
		return nil, ctxerr.With(fmt.Errorf("walking models dir %s: %w", root, err), map[string]any{"dir": root})
	}
	return &gatewaypb.GatewayListGroupsResponse{Groups: groupModels(infos)}, nil
}

// planHFInstall resolves an HF install pick into primary + mmproj
// companion (when present) and reserves both destination paths in the
// catalogue so a concurrent walk doesn't drop them as orphans
// pre-download. Called from handlers.go::handleInstallSubmit, shipped
// to MASS via MassScheduler.DownloadFiles.
//
// Companion is matched by filename pre-download (only signal available);
// the projector's GGUF header (architecture=clip) confirms at walk time.
func planHFInstall(ctx context.Context, modelsDir string, cache *metadataCache, repoID, filename, groupName string) ([]*gatewaypb.DownloadFile, error) {
	repoID = strings.TrimSpace(repoID)
	filename = strings.TrimSpace(filename)
	groupName = strings.TrimSpace(groupName)
	if repoID == "" || filename == "" {
		return nil, fmt.Errorf("repo_id and filename are required")
	}
	if groupName == "" {
		return nil, fmt.Errorf("group_name is required")
	}

	files, err := hf.ListFiles(ctx, repoID, []string{".gguf"})
	if err != nil {
		return nil, ctxerr.With(fmt.Errorf("listing HF files: %w", err), map[string]any{"repo_id": repoID})
	}

	var primarySize int64 = -1
	for _, f := range files {
		if f.Filename == filename {
			primarySize = f.SizeBytes
			break
		}
	}

	// Only auto-attach an mmproj companion when the user picked a
	// non-projector primary. Picking the projector itself is an
	// explicit single-file install and shouldn't pull siblings.
	var mmprojName string
	var mmprojSize int64 = -1
	if !looksLikeMmprojFilename(filename) {
		mmprojName, mmprojSize = pickMmprojCompanion(files, filename)
	}
	if primarySize < 0 {
		return nil, ctxerr.With(fmt.Errorf("file %q not found in repo %q", filename, repoID), map[string]any{"repo_id": repoID, "filename": filename})
	}

	primaryDest := cache.groupRelPath(groupName, filename)
	if err := cache.reserveEntry(filepath.Join(modelsDir, primaryDest), groupName); err != nil {
		return nil, reserveErrToStatus(err)
	}

	out := []*gatewaypb.DownloadFile{
		{
			Url:        hfFileURL(repoID, filename),
			RelPath:    primaryDest,
			SizeBytes:  primarySize,
			GroupLabel: groupName,
		},
	}
	if mmprojName != "" {
		mmprojDest := cache.groupRelPath(groupName, mmprojName)
		if err := cache.reserveEntry(filepath.Join(modelsDir, mmprojDest), groupName); err != nil {
			return nil, reserveErrToStatus(err)
		}
		out = append(out, &gatewaypb.DownloadFile{
			Url:        hfFileURL(repoID, mmprojName),
			RelPath:    mmprojDest,
			SizeBytes:  mmprojSize,
			GroupLabel: groupName,
		})
	}
	return out, nil
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
	modelsDir, cache, ok := g.snapshot()
	if !ok {
		return nil, errNotInitialised("plan_local_import")
	}
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

	dest := cache.groupRelPath(groupName, src)
	if err := cache.reserveEntry(filepath.Join(modelsDir, dest), groupName); err != nil {
		return nil, reserveErrToStatus(err)
	}

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

// PlanRemoteImport plans an HF install: it resolves repo+filename into the
// primary plus any mmproj companion the gateway recognises, and returns the
// [DownloadFile] list MASS fetches under models_dir. Pure decision — the
// gateway never downloads. Mirrors PlanLocalImport for the remote case and
// reuses the same resolver as the operator install UI.
func (g *Gateway) PlanRemoteImport(ctx context.Context, req *gatewaypb.PlanRemoteImportRequest) (*gatewaypb.PlanRemoteImportResponse, error) {
	modelsDir, cache, ok := g.snapshot()
	if !ok {
		return nil, errNotInitialised("plan_remote_import")
	}
	files, err := planHFInstall(ctx, modelsDir, cache, req.GetRepoId(), req.GetFilename(), req.GetGroupName())
	if err != nil {
		// planHFInstall already returns a gRPC status (e.g. AlreadyExists
		// when the destination is taken) for the cases callers act on;
		// pass those through and default the rest to InvalidArgument.
		if _, ok := status.FromError(err); ok {
			return nil, err
		}
		return nil, status.Errorf(codes.InvalidArgument, "plan_remote_import: %v", err)
	}
	return &gatewaypb.PlanRemoteImportResponse{Files: files}, nil
}

// PlanDelete resolves a model id (Group.id slug) to the store-relative files
// that constitute it — the primary plus any companions sharing its Name. The
// gateway only decides; MASS removes the files and the next catalogue walk
// prunes the entries. Returns NotFound when no group matches id.
//
// Residency is enforced before the plan is handed back: if any loaded
// instance backed by a doomed file is still serving (active > 0) the
// delete is refused with FailedPrecondition — removing bytes out from
// under in-flight requests would fail them. Idle residents are evicted
// (via MASS) so no worker keeps an OS lock on files about to vanish.
func (g *Gateway) PlanDelete(ctx context.Context, req *gatewaypb.PlanDeleteRequest) (*gatewaypb.PlanDeleteResponse, error) {
	_, cache, ok := g.snapshot()
	if !ok {
		return nil, errNotInitialised("plan_delete")
	}
	id := strings.TrimSpace(req.GetId())
	if id == "" {
		return nil, status.Error(codes.InvalidArgument, "plan_delete: id is required")
	}
	relPaths := cache.relPathsForModel(id)
	if len(relPaths) == 0 {
		// Delete is idempotent: a group's variants are deleted one id at a
		// time, but the first delete removes every file sharing the Name
		// (primary + companions), so the later ids resolve to nothing. A
		// published-id-shaped target that's already gone is a no-op success,
		// not a failure; only a malformed id is NotFound.
		if isPublishedID(id) {
			return &gatewaypb.PlanDeleteResponse{}, nil
		}
		return nil, status.Errorf(codes.NotFound, "plan_delete: no model matches id %q", id)
	}
	if err := g.evictLoadedInstances(ctx, relPaths); err != nil {
		return nil, err
	}
	return &gatewaypb.PlanDeleteResponse{RelPaths: relPaths}, nil
}

// evictLoadedInstances refuses the delete if any instance backed by
// relPaths is serving, then evicts the idle ones via MASS before the
// files are removed. relPaths are FULL store-relative keys (with the
// runtime-owned first segment, "gguf/…").
//
// Returns FailedPrecondition when a matching instance has active jobs.
// Eviction of idle instances is best-effort: failures are logged and the
// plan proceeds — MASS's file removal surfaces its own error if a worker
// still holds the file open. A nil scheduler (not yet wired) skips both.
func (g *Gateway) evictLoadedInstances(ctx context.Context, relPaths []string) error {
	g.mu.RLock()
	scheduler := g.scheduler
	g.mu.RUnlock()
	if scheduler == nil {
		return nil
	}
	workers, err := scheduler.ListWorkers(ctx)
	if err != nil {
		g.logger.Warn().Err(err).Msg("listing workers before delete; skipping eviction")
		return nil
	}
	instances := loadedInstancesForRelPaths(workers, relPaths)
	var active int32
	for _, in := range instances {
		active += in.active
	}
	if active > 0 {
		return status.Errorf(codes.FailedPrecondition, "plan_delete: model is serving %d active job(s); cancel or wait for them before deleting", active)
	}
	for _, in := range instances {
		n, err := scheduler.EvictModel(ctx, in.modelID, "")
		if err != nil {
			g.logger.Warn().Err(err).Str("model_id", in.modelID).Msg("evicting loaded model before delete")
			continue
		}
		g.logger.Info().Str("model_id", in.modelID).Int32("evicted", n).Msg("evicted loaded model before delete")
	}
	return nil
}

// loadedInstance is one loaded model instance matched to a doomed group:
// its opaque model_id (used to evict) and its in-flight job count (used
// to refuse deletes that would fail active requests).
type loadedInstance struct {
	modelID string
	active  int32
}

// loadedInstancesForRelPaths returns the distinct loaded instances whose
// backing files are among relPaths (FULL store-relative keys). Matching
// is files-first: a worker reports LoadedModelStatus.files — the same
// store-relative cache keys as ModelFile.filename / relPaths — so a
// doomed path matches a loaded model when it equals a reported file OR
// sits under one (or vice versa), honouring the dir-subtree contract
// where a files entry may denote a directory root.
//
// Only when a worker reports NO files for an instance (an older worker,
// or one that hasn't populated the field) do we fall back to parsing the
// model_id's relpath prefix. That relpath is the request-facing model
// string, which omits the runtime-owned first segment, so relPaths are
// stripped of formatDir before comparing.
func loadedInstancesForRelPaths(workers []*gatewaypb.WorkerSummary, relPaths []string) []loadedInstance {
	fallbackWant := make(map[string]struct{}, len(relPaths))
	for _, p := range relPaths {
		fallbackWant[strings.TrimPrefix(p, formatDir+"/")] = struct{}{}
	}
	var out []loadedInstance
	seen := map[string]struct{}{}
	for _, w := range workers {
		for _, lm := range w.GetLoadedModels() {
			id := lm.GetModelId()
			files := lm.GetFiles()
			var hit bool
			if len(files) > 0 {
				hit = anyPathMatches(relPaths, files)
			} else {
				rel, _, _ := strings.Cut(id, "#")
				_, hit = fallbackWant[rel]
			}
			if !hit {
				continue
			}
			if _, dup := seen[id]; dup {
				continue
			}
			seen[id] = struct{}{}
			out = append(out, loadedInstance{modelID: id, active: lm.GetActive()})
		}
	}
	return out
}

// anyPathMatches reports whether any doomed path overlaps any backing
// file under the dir-subtree contract: two forward-slash store keys
// overlap when they are equal or one is a directory ancestor of the
// other (prefix + "/"). Either side may be the subtree root.
func anyPathMatches(relPaths, files []string) bool {
	for _, rp := range relPaths {
		for _, f := range files {
			if rp == f ||
				strings.HasPrefix(f, rp+"/") ||
				strings.HasPrefix(rp, f+"/") {
				return true
			}
		}
	}
	return false
}

// RenameGroup retags every catalogue entry whose Name slug matches id
// with new_name. Returns NotFound when no entry matches.
//
// Rename is catalogue-only: files never move, so loaded workers and
// in-flight jobs (keyed on the physical store path) are unaffected.
// Published model ids re-derive from the new name on the next list;
// ids handed out before the rename keep resolving through
// storePathForID.
func (g *Gateway) RenameGroup(_ context.Context, req *gatewaypb.RenameGroupRequest) (*gatewaypb.RenameGroupResponse, error) {
	_, cache, ok := g.snapshot()
	if !ok {
		return nil, errNotInitialised("rename_group")
	}
	id := strings.TrimSpace(req.GetId())
	newName := strings.TrimSpace(req.GetNewName())
	if id == "" || newName == "" {
		return nil, status.Error(codes.InvalidArgument, "rename_group: id and new_name are required")
	}
	old := cache.nameForSlug(id)
	if old == "" {
		return nil, status.Errorf(codes.NotFound, "rename_group: no group matches id %q", id)
	}
	if old == newName {
		return &gatewaypb.RenameGroupResponse{}, nil
	}
	n, err := cache.renameGroup(old, newName)
	if err != nil {
		if errors.Is(err, errNameNotSluggable) || errors.Is(err, errSlugCollision) {
			return nil, reserveErrToStatus(err)
		}
		return nil, ctxerr.With(fmt.Errorf("renaming group %q → %q: %w", old, newName, err), map[string]any{"old": old, "new": newName})
	}
	if n == 0 {
		return nil, status.Errorf(codes.NotFound, "rename_group: no entries renamed for %q", old)
	}
	cache.saveToDisk()
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

// reserveErrToStatus maps reserveEntry/renameGroup identity failures
// to the gRPC codes MASS surfaces to the operator: a name that can't
// slug at all is a bad argument; a slug collision or an occupied
// destination conflicts with existing state. Anything else passes
// through unchanged.
func reserveErrToStatus(err error) error {
	switch {
	case errors.Is(err, errNameNotSluggable):
		return status.Error(codes.InvalidArgument, err.Error())
	case errors.Is(err, errSlugCollision), errors.Is(err, errDestTaken):
		return status.Error(codes.AlreadyExists, err.Error())
	default:
		return err
	}
}

// hfFileURL is the canonical resolve URL for a file in a HuggingFace
// repo. filename may live in a subfolder, so each path segment is
// escaped separately — escaping the whole string would turn the "/"
// separators into %2F and break the resolve path.
func hfFileURL(repoID, filename string) string {
	segments := strings.Split(filename, "/")
	for i, s := range segments {
		segments[i] = url.PathEscape(s)
	}
	return "https://huggingface.co/" + repoID + "/resolve/main/" + strings.Join(segments, "/")
}

// pickMmprojCompanion selects the mmproj sibling to bundle with a
// non-projector primary. Repos often ship several precisions
// (mmproj-*-f16 and mmproj-*-f32), and the API's listing order isn't a
// contract — prefer the f16 variant deterministically and fall back to
// the first candidate when no f16 variant exists.
func pickMmprojCompanion(files []hf.GGUFFile, primary string) (string, int64) {
	name := ""
	var size int64 = -1
	for _, f := range files {
		if f.Filename == primary || !looksLikeMmprojFilename(f.Filename) {
			continue
		}
		if strings.Contains(strings.ToLower(f.Filename), "f16") {
			return f.Filename, f.SizeBytes
		}
		if name == "" {
			name = f.Filename
			size = f.SizeBytes
		}
	}
	return name, size
}

// looksLikeMmprojFilename is the pre-download filename heuristic for
// spotting a projector. Used by PlanModelFiles + FilterFilenames where
// no header is available yet; identity is operator-typed, this is only
// a fetch-time bundling hint.
func looksLikeMmprojFilename(name string) bool {
	lower := strings.ToLower(name)
	return strings.Contains(lower, "mmproj") && strings.HasSuffix(lower, ".gguf")
}

// capabilitiesFromHeader derives Capabilities from header signals only.
// Vision is set at walk time by sibling-projector lookup (primary's
// own header doesn't expose it). Audio: not inferable from llama.cpp
// GGUFs. Thinking: chat-template substring, gated on model_type=chat
// because some embedding models (Qwen3-Embedding) ship think-token
// chat templates but don't emit reasoning.
func capabilitiesFromHeader(kv map[string]string) *gatewaypb.Capabilities {
	thinking := strings.EqualFold(kv["thinking"], "true") && modelTypeFromHeader(kv) == "chat"
	return &gatewaypb.Capabilities{
		Thinking: thinking,
	}
}

// modelTypeFromHeader distinguishes chat vs embedding. clip arch
// returns "" (not standalone). pooling_type wins — some embedding
// models (Qwen3-Embedding) ship a chat template, so template-presence
// alone misclassifies them. Falls back to chat_template_present for
// loosely-tagged older GGUFs.
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

func extOf(path string) string {
	i := strings.LastIndexByte(path, '.')
	if i < 0 || i == len(path)-1 {
		return ""
	}
	return path[i+1:]
}
