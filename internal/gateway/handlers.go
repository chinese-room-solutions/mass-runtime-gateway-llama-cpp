package gateway

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path"
	"path/filepath"

	"github.com/KernelPryanic/ctxerr"
	gatewaypb "github.com/chinese-room-solutions/mass-proto/gen/go/gateway"
	llamacpp "github.com/chinese-room-solutions/mass-proto/gen/go/llama-cpp"
	workerpb "github.com/chinese-room-solutions/mass-proto/gen/go/worker"
	"github.com/chinese-room-solutions/mass-runtime-gateway-llama-cpp/internal/model"
	"github.com/chinese-room-solutions/mass-runtime-gateway-llama-cpp/internal/payload"
	"github.com/chinese-room-solutions/mass-runtime-gateway-llama-cpp/internal/sched"
	"github.com/rs/zerolog"
)

// handlers holds the typed-API and OpenAI-compat HTTP handlers. Constructed
// once per Gateway init.
type handlers struct {
	params      PluginParams
	runtimeName string
	modelsDir   string
	scheduler   *sched.Client
	cache       *metadataCache
	logger      zerolog.Logger
}

func newHandlers(d routerDeps) *handlers {
	return &handlers{
		params:      d.params,
		runtimeName: d.runtimeName,
		modelsDir:   d.modelsDir,
		scheduler:   d.scheduler,
		cache:       d.cache,
		logger:      d.logger.With().Str("component", "handlers").Logger(),
	}
}

// ----- Request shapes (typed API) -----

// ChatRequest is the gateway-typed body for Chat / ChatStream / BatchChat.
type chatRequest struct {
	Model    string          `json:"model"`
	Config   *loadConfig     `json:"config,omitempty"` // optional; per-request load overrides
	Messages []chatMessage   `json:"messages"`
	Sampling *samplingParams `json:"sampling,omitempty"`
}

type batchChatRequest struct {
	Model  string          `json:"model"`
	Config *loadConfig     `json:"config,omitempty"`
	Items  []batchChatItem `json:"items"`
}

type batchChatItem struct {
	Messages []chatMessage   `json:"messages"`
	Sampling *samplingParams `json:"sampling,omitempty"`
}

type embedRequest struct {
	Model  string      `json:"model"`
	Config *loadConfig `json:"config,omitempty"`
	Input  string      `json:"input"`
}

type batchEmbedRequest struct {
	Model  string      `json:"model"`
	Config *loadConfig `json:"config,omitempty"`
	Inputs []string    `json:"inputs"`
}

type tokenizeRequest struct {
	Model  string      `json:"model"`
	Config *loadConfig `json:"config,omitempty"`
	Text   string      `json:"text"`
}

// loadConfig mirrors llamacpp.LoadHints in JSON form. Pointer fields use
// optional-style JSON ("set" iff present in the body).
type loadConfig struct {
	ContextSize    int32  `json:"context_size,omitempty"`
	BatchSize      *int32 `json:"batch_size,omitempty"`
	GPULayers      *int32 `json:"gpu_layers,omitempty"`
	FlashAttn      *bool  `json:"flash_attn,omitempty"`
	Threads        *int32 `json:"threads,omitempty"`
	MaxConcurrent  *int32 `json:"max_concurrent,omitempty"`
	Thinking       bool   `json:"thinking,omitempty"`
	MmprojFilename string `json:"mmproj_filename,omitempty"`
	ChatTemplate   string `json:"chat_template,omitempty"`
	CacheType      string `json:"cache_type,omitempty"` // "f16" | "q8_0" | "q4_0"
	// VRAM headroom watermark, 1-100. Worker stops growing the per-model
	// context pool when any allowed GPU's used/total ratio crosses this %.
	// nil → use the worker's --vram-headroom-pct default.
	VramHeadroomPct *int32 `json:"vram_headroom_pct,omitempty"`
}

type chatMessage struct {
	Role    string        `json:"role"`
	Content string        `json:"content,omitempty"`
	Parts   []contentPart `json:"parts,omitempty"`
}

type contentPart struct {
	Type     string `json:"type"` // "text" | "image" | "audio"
	Text     string `json:"text,omitempty"`
	Data     []byte `json:"data,omitempty"`
	MIMEType string `json:"mime_type,omitempty"`
}

// samplingParams mirrors llamacpp.SamplingParams in JSON form. Numeric
// fields are pointers so JSON presence survives to the wire: a field the
// client omitted stays absent (worker default applies), while an explicit
// zero passes through as present (temperature:0 → greedy, seed:0 → seed 0).
type samplingParams struct {
	MaxTokens        *int32   `json:"max_tokens,omitempty"`
	Temperature      *float32 `json:"temperature,omitempty"`
	TopP             *float32 `json:"top_p,omitempty"`
	TopK             *int32   `json:"top_k,omitempty"`
	Seed             *int32   `json:"seed,omitempty"`
	Stop             []string `json:"stop,omitempty"`
	MinP             *float32 `json:"min_p,omitempty"`
	RepeatPenalty    *float32 `json:"repeat_penalty,omitempty"`
	FrequencyPenalty *float32 `json:"frequency_penalty,omitempty"`
	PresencePenalty  *float32 `json:"presence_penalty,omitempty"`
	EnableThinking   bool     `json:"enable_thinking,omitempty"`
}

// ----- Response shapes (typed API) -----

type chatResponse struct {
	ID               string       `json:"id"`
	Model            string       `json:"model"`
	Message          *chatMessage `json:"message,omitempty"`
	FinishReason     string       `json:"finish_reason,omitempty"`
	ReasoningContent string       `json:"reasoning_content,omitempty"`
	Usage            *usage       `json:"usage,omitempty"`
	TokensPerSecond  float64      `json:"tokens_per_second,omitempty"`
}

type usage struct {
	PromptTokens     int32 `json:"prompt_tokens"`
	CompletionTokens int32 `json:"completion_tokens"`
	TotalTokens      int32 `json:"total_tokens"`
}

type embedResponse struct {
	ID        string    `json:"id"`
	Model     string    `json:"model"`
	Embedding []float32 `json:"embedding"`
}

type batchEmbedResponse struct {
	ID         string          `json:"id"`
	Model      string          `json:"model"`
	Embeddings []embeddingItem `json:"embeddings"`
}

type embeddingItem struct {
	Index     int32     `json:"index"`
	Embedding []float32 `json:"embedding"`
}

type tokenizeResponse struct {
	Tokens []int32 `json:"tokens"`
}

// ----- Handlers -----

// submitResponse is the body every typed POST returns: just the job id. The
// caller then reads the result via GET /.v1/Jobs/{id} (block-until-terminal or
// poll) and cancels via DELETE /.v1/Jobs/{id}.
type submitResponse struct {
	JobID string `json:"job_id"`
}

// submit enqueues a job and returns its id immediately. This is the only place
// a typed POST does work: build → SubmitOnly → {job_id}. Pre-schedule failures
// (bad request, load/dispatch error) carry their HTTP status via the build
// closure / statusError, so submission errors are honest 4xx/5xx. Once the id
// is returned the job is durable; only DELETE /Jobs/{id} stops it.
func (h *handlers) submit(w http.ResponseWriter, r *http.Request, build buildJobFunc) {
	job, modelID, hints, files, err := build()
	if err != nil {
		writeStatusError(w, err)
		return
	}
	params, err := h.buildScheduleParams(r.Context(), modelID, job, hints, files)
	if err != nil {
		writeStatusError(w, err)
		return
	}
	jobID, err := h.scheduler.SubmitOnly(r.Context(), params)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	w.Header().Set(jobIDHeader, jobID)
	writeJSON(w, http.StatusOK, submitResponse{JobID: jobID})
}

func (h *handlers) handleChat(w http.ResponseWriter, r *http.Request) {
	h.submit(w, r, func() (*llamacpp.Job, string, *llamacpp.LoadHints, []*workerpb.ModelFile, error) {
		var req chatRequest
		if err := readJSON(r, &req); err != nil {
			return nil, "", nil, nil, badRequest(err)
		}
		job, modelID, err := h.buildChatJob(&req, false)
		if err != nil {
			return nil, "", nil, nil, badRequest(err)
		}
		hints, files, err := h.buildLoadArtifacts(req.Model, req.Config, llamacpp.LoadKind_LOAD_KIND_CHAT)
		if err != nil {
			return nil, "", nil, nil, badRequest(err)
		}
		return job, modelID, hints, files, nil
	})
}

// jobResultResponse is the body returned when polling a job. Status is one of
// "pending" | "processing" | "done" | "error". Result is set only when done
// (its shape depends on the job type — chat/embed/batch-embed/tokenize); Error
// only when status is "error".
type jobResultResponse struct {
	JobID  string          `json:"job_id"`
	Status string          `json:"status"`
	Result json.RawMessage `json:"result,omitempty"`
	Error  string          `json:"error,omitempty"`
}

// resolveStorePath turns a request's "model" string into the physical
// under-format-root store path every downstream consumer (load
// artifacts, scheduler model id, detail lookup) keys on. Published ids
// carry the current group slug, which after a group rename no longer
// matches the on-disk directory — the catalogue maps them back.
// Physical paths and uncatalogued files pass through unchanged, so
// pre-rename model strings keep working. Returns an error when the
// path is invalid (absolute, ".." traversal, empty).
func (h *handlers) resolveStorePath(modelStr string) (string, error) {
	storePath := model.ResolveModelPath(modelStr)
	if storePath == "" {
		return "", fmt.Errorf("model: invalid path %q", modelStr)
	}
	if h.cache != nil {
		storePath = h.cache.storePathForID(storePath)
	}
	return storePath, nil
}

// handleJobResult reads a submitted job's durable result by id. With ?wait=1 it
// blocks until the job reaches a terminal state (done|error), then returns it;
// without it, it returns the current status immediately (poll). This endpoint
// is read-only: dropping the connection NEVER cancels the job — it stays
// durable, reconnect or poll later. Cancel is DELETE /Jobs/{id}. The stored
// payload self-describes its type via the JobChunk oneof, so one endpoint
// serves chat, embed, batch-embed, and tokenize results.
func (h *handlers) handleJobResult(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	if id == "" {
		writeError(w, http.StatusBadRequest, fmt.Errorf("job id required"))
		return
	}
	if r.URL.Query().Get("wait") == "1" {
		// Block until terminal by draining the reattach stream. The job is
		// durable in MASS; this is purely a read. A client disconnect cancels
		// r.Context() and ends the drain — the job keeps running.
		for range h.scheduler.Reattach(r.Context(), id) { //nolint:revive // drain to terminal; result read below
		}
		if r.Context().Err() != nil {
			return // client went away mid-wait; nothing to write
		}
	}
	res, err := h.scheduler.GetResult(r.Context(), id)
	if err != nil {
		if errors.Is(err, sched.ErrResultNotFound) {
			writeError(w, http.StatusNotFound, fmt.Errorf("job %q not found or expired", id))
			return
		}
		writeError(w, http.StatusInternalServerError, err)
		return
	}

	out := jobResultResponse{JobID: id, Status: jobStatusString(res.Status)}
	switch res.Status {
	case sched.ResultDone:
		result, decErr := decodeJobResult(id, res.Body)
		if decErr != nil {
			writeError(w, http.StatusInternalServerError, decErr)
			return
		}
		out.Result = result
	case sched.ResultError:
		out.Error = res.Err
	}
	writeJSON(w, http.StatusOK, out)
}

// decodeJobResult turns a stored terminal JobChunk into the JSON response shape
// matching its job type. Switches on the JobChunk oneof — the same per-type
// decode the synchronous handlers do, collapsed into one place.
func decodeJobResult(id string, body []byte) (json.RawMessage, error) {
	if len(body) == 0 {
		return nil, nil
	}
	dec, err := payload.DecodeJobChunk(body)
	if err != nil {
		return nil, ctxerr.With(fmt.Errorf("decoding job result: %w", err), map[string]any{"job_id": id})
	}
	switch {
	case dec.GetChatFinal() != nil:
		resp := &chatResponse{ID: id}
		applyChatFinal(resp, dec.GetChatFinal())
		return json.Marshal(resp)
	case dec.GetEmbed() != nil:
		return json.Marshal(embedResponse{ID: id, Embedding: dec.GetEmbed().GetEmbedding()})
	case dec.GetBatchEmbed() != nil:
		br := dec.GetBatchEmbed()
		resp := batchEmbedResponse{ID: id}
		resp.Embeddings = make([]embeddingItem, len(br.GetItems()))
		for i, it := range br.GetItems() {
			resp.Embeddings[i] = embeddingItem{Index: it.GetIndex(), Embedding: it.GetEmbedding()}
		}
		return json.Marshal(resp)
	case dec.GetTokenize() != nil:
		return json.Marshal(tokenizeResponse{Tokens: dec.GetTokenize().GetTokens()})
	case dec.GetBatchChat() != nil:
		br := dec.GetBatchChat()
		return json.Marshal(batchChatResponse{ID: id, Responses: batchChatFinalsToResponses(br)})
	default:
		return nil, nil
	}
}

// handleJobCancel cancels a submitted job (pending or running) by id.
func (h *handlers) handleJobCancel(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	if id == "" {
		writeError(w, http.StatusBadRequest, fmt.Errorf("job id required"))
		return
	}
	if err := h.scheduler.CancelJob(r.Context(), id); err != nil {
		if errors.Is(err, sched.ErrResultNotFound) {
			writeError(w, http.StatusNotFound, fmt.Errorf("job %q not cancellable (finished, unknown, or expired)", id))
			return
		}
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

func jobStatusString(s sched.ResultStatus) string {
	switch s {
	case sched.ResultProcessing:
		return "processing"
	case sched.ResultDone:
		return "done"
	case sched.ResultError:
		return "error"
	default:
		return "pending"
	}
}

// batchChatResponse is the body returned by BatchChat (and when fetching a
// stored BatchChat result by id): one chat response per input item,
// index-aligned.
type batchChatResponse struct {
	ID        string         `json:"id"`
	Model     string         `json:"model,omitempty"`
	Responses []chatResponse `json:"responses"`
}

func (h *handlers) handleBatchChat(w http.ResponseWriter, r *http.Request) {
	h.submit(w, r, func() (*llamacpp.Job, string, *llamacpp.LoadHints, []*workerpb.ModelFile, error) {
		var req batchChatRequest
		if err := readJSON(r, &req); err != nil {
			return nil, "", nil, nil, badRequest(err)
		}
		return h.buildBatchChatJob(&req)
	})
}

// buildBatchChatJob assembles the single BatchChatJob, model ID, and load
// artifacts shared by the sync and async batch-chat handlers.
func (h *handlers) buildBatchChatJob(req *batchChatRequest) (*llamacpp.Job, string, *llamacpp.LoadHints, []*workerpb.ModelFile, error) {
	if len(req.Items) == 0 {
		return nil, "", nil, nil, fmt.Errorf("items must be non-empty")
	}
	hints, files, err := h.buildLoadArtifacts(req.Model, req.Config, llamacpp.LoadKind_LOAD_KIND_CHAT)
	if err != nil {
		return nil, "", nil, nil, err
	}
	storePath, err := h.resolveStorePath(req.Model)
	if err != nil {
		return nil, "", nil, nil, err
	}
	items := make([]*llamacpp.BatchChatItem, len(req.Items))
	for i, it := range req.Items {
		items[i] = &llamacpp.BatchChatItem{
			Messages: chatMessagesToProto(it.Messages),
			Sampling: convertSampling(it.Sampling),
		}
	}
	job := &llamacpp.Job{
		Kind: llamacpp.JobKind_JOB_KIND_BATCH_CHAT,
		Body: &llamacpp.Job_BatchChat{BatchChat: &llamacpp.BatchChatJob{Items: items}},
	}
	return job, model.ID(storePath, hints), hints, files, nil
}

// batchChatFinalsToResponses maps a BatchChatResult into the index-aligned
// chat responses the HTTP shape exposes.
func batchChatFinalsToResponses(br *llamacpp.BatchChatResult) []chatResponse {
	out := make([]chatResponse, len(br.GetItems()))
	for i, cf := range br.GetItems() {
		out[i] = chatResponse{ID: cf.GetId()}
		applyChatFinal(&out[i], cf)
	}
	return out
}

// simpleJobBuilder returns a buildJobFunc for a single-shot job (embed /
// batch-embed / tokenize): resolve load artifacts + model id for modelStr,
// pair them with the prebuilt job.
func (h *handlers) simpleJobBuilder(modelStr string, cfg *loadConfig, kind llamacpp.LoadKind, job *llamacpp.Job) buildJobFunc {
	return func() (*llamacpp.Job, string, *llamacpp.LoadHints, []*workerpb.ModelFile, error) {
		storePath, err := h.resolveStorePath(modelStr)
		if err != nil {
			return nil, "", nil, nil, badRequest(err)
		}
		hints, files, err := h.buildLoadArtifacts(modelStr, cfg, kind)
		if err != nil {
			return nil, "", nil, nil, badRequest(err)
		}
		return job, model.ID(storePath, hints), hints, files, nil
	}
}

func (h *handlers) handleEmbed(w http.ResponseWriter, r *http.Request) {
	h.submit(w, r, func() (*llamacpp.Job, string, *llamacpp.LoadHints, []*workerpb.ModelFile, error) {
		var req embedRequest
		if err := readJSON(r, &req); err != nil {
			return nil, "", nil, nil, badRequest(err)
		}
		job := &llamacpp.Job{
			Kind: llamacpp.JobKind_JOB_KIND_EMBED,
			Body: &llamacpp.Job_Embed{Embed: &llamacpp.EmbedJob{Input: req.Input}},
		}
		return h.simpleJobBuilder(req.Model, req.Config, llamacpp.LoadKind_LOAD_KIND_EMBEDDING, job)()
	})
}

func (h *handlers) handleBatchEmbed(w http.ResponseWriter, r *http.Request) {
	h.submit(w, r, func() (*llamacpp.Job, string, *llamacpp.LoadHints, []*workerpb.ModelFile, error) {
		var req batchEmbedRequest
		if err := readJSON(r, &req); err != nil {
			return nil, "", nil, nil, badRequest(err)
		}
		job := &llamacpp.Job{
			Kind: llamacpp.JobKind_JOB_KIND_BATCH_EMBED,
			Body: &llamacpp.Job_BatchEmbed{BatchEmbed: &llamacpp.BatchEmbedJob{Inputs: req.Inputs}},
		}
		return h.simpleJobBuilder(req.Model, req.Config, llamacpp.LoadKind_LOAD_KIND_EMBEDDING, job)()
	})
}

func (h *handlers) handleTokenize(w http.ResponseWriter, r *http.Request) {
	h.submit(w, r, func() (*llamacpp.Job, string, *llamacpp.LoadHints, []*workerpb.ModelFile, error) {
		var req tokenizeRequest
		if err := readJSON(r, &req); err != nil {
			return nil, "", nil, nil, badRequest(err)
		}
		job := &llamacpp.Job{
			Kind: llamacpp.JobKind_JOB_KIND_TOKENIZE,
			Body: &llamacpp.Job_Tokenize{Tokenize: &llamacpp.TokenizeJob{Text: req.Text}},
		}
		// Tokenize uses the chat tokenizer in this gateway.
		return h.simpleJobBuilder(req.Model, req.Config, llamacpp.LoadKind_LOAD_KIND_CHAT, job)()
	})
}

func (h *handlers) handleListModels(w http.ResponseWriter, r *http.Request) {
	_ = r
	infos, err := h.cache.walkAndParseModels(formatRoot(h.modelsDir))
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	out := make([]map[string]any, 0, len(infos))
	for _, info := range infos {
		if info.Companion != "" {
			continue
		}
		out = append(out, map[string]any{
			"id":         info.ID,
			"size_bytes": info.SizeBytes,
		})
	}
	writeJSON(w, http.StatusOK, out)
}

// handleModelDetailHTML returns the rendered HTML props panel for a single
// model. Looks up the file via the same shared cache as the list view and
// renders via [renderModelDetail]. Returns an empty body when the id is
// missing or the file isn't recognised — MASS leaves the right pane blank
// in that case instead of surfacing an error.
func (h *handlers) handleModelDetailHTML(w http.ResponseWriter, r *http.Request) {
	id := r.URL.Query().Get("id")
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	if id == "" {
		return
	}
	// Published ids carry the current group slug; map back to the
	// physical store path before touching the disk.
	storeID := h.cache.storePathForID(id)
	abs := absForStoreID(h.modelsDir, storeID)
	info, ok := h.cache.parseModelInfo(abs, storeID)
	if !ok {
		return
	}
	if _, err := io.WriteString(w, renderModelDetail(h.runtimeName, info)); err != nil {
		h.logger.Debug().Err(err).Msg("writing model detail html")
	}
}

// ----- Shared helpers -----

// dispatch is the common path for inference: ship the job payload plus the
// load artifacts (files + hints + source) MASS may need if the chosen
// worker doesn't already have modelID resident. MASS load-on-demands at
// dispatch using these, so there is no preload preamble.
// jobIDHeader carries a job's request_id back to the caller on every typed
// request, set before the body streams. A caller that disconnects can fetch
// the durable result later via GET /.v1/Jobs/{id}.
const jobIDHeader = "X-Mass-Job-Id"

// dispatchWithID submits the job and returns its request_id plus the live
// chunk stream. The job is durable in MASS regardless of whether the caller
// drains the stream. Used by the streaming gRPC and OpenAI-compat paths.
func (h *handlers) dispatchWithID(ctx context.Context, modelID string, job *llamacpp.Job, hints *llamacpp.LoadHints, files []*workerpb.ModelFile) (string, <-chan sched.JobChunk, error) {
	params, err := h.buildScheduleParams(ctx, modelID, job, hints, files)
	if err != nil {
		return "", nil, err
	}
	return h.scheduler.ScheduleWithID(ctx, params)
}

// decodeWarner returns a per-stream closure that logs the FIRST
// malformed worker chunk and swallows the rest. Dropping a chunk the
// gateway can't decode is the only sane recovery mid-stream, but doing
// it silently hides a misbehaving worker; one warning per job keeps it
// visible without flooding the log on a garbage stream.
func (h *handlers) decodeWarner(jobID string) func(error) {
	warned := false
	return func(err error) {
		if warned {
			return
		}
		warned = true
		h.logger.Warn().Err(err).Str("job_id", jobID).
			Msg("dropping undecodable worker chunk; suppressing further decode warnings for this job")
	}
}

// buildJobFunc parses a typed request and assembles its job + model id + load
// artifacts. Returned errors should already carry their HTTP status (e.g. via
// [badRequest]).
type buildJobFunc func() (*llamacpp.Job, string, *llamacpp.LoadHints, []*workerpb.ModelFile, error)

// buildScheduleParams encodes the job + load artifacts and predicts the
// cost/memory fields MASS scores on. Shared by the sync and async paths.
func (h *handlers) buildScheduleParams(ctx context.Context, modelID string, job *llamacpp.Job, hints *llamacpp.LoadHints, files []*workerpb.ModelFile) (sched.ScheduleParams, error) {
	bodyBytes, err := payload.EncodeJob(job)
	if err != nil {
		return sched.ScheduleParams{}, err
	}
	hintsBytes, err := payload.EncodeLoadHints(hints)
	if err != nil {
		return sched.ScheduleParams{}, ctxerr.With(fmt.Errorf("encoding load hints: %w", err), map[string]any{"model_id": modelID})
	}
	cost, axis := predictCost(job, h.primaryParameterCount(files), h.primaryThinking(files), h.visionParamsFor(files))
	base, perSlot, headroom := h.loadByteEstimate(files, hints)
	// Batch work is throughput-oriented and must not delay interactive
	// chat on the same worker queue: submit it at LOW priority.
	var prio gatewaypb.JobPriority
	switch job.GetKind() {
	case llamacpp.JobKind_JOB_KIND_BATCH_CHAT, llamacpp.JobKind_JOB_KIND_BATCH_EMBED:
		prio = gatewaypb.JobPriority_JOB_PRIORITY_LOW
	}
	return sched.ScheduleParams{
		ModelID:       modelID,
		Payload:       bodyBytes,
		Cost:          cost,
		CostAxis:      axis,
		Files:         files,
		LoadHints:     hintsBytes,
		BaseLoadBytes: base,
		PerSlotBytes:  perSlot,
		HeadroomPct:   headroom,
		Source:        sourceFromContext(ctx),
		Priority:      prio,
	}, nil
}

// primaryParameterCount looks up the compute-relevant parameter count
// (per-token active for MoE, total for dense — see
// [gguf.Meta.ActiveParameterCount]) for the dispatch's primary model
// file from the metadata cache. Returns 0 when the file isn't
// catalogued or the header didn't carry a usable count; [predictCost]
// then falls back to a median model size.
func (h *handlers) primaryParameterCount(files []*workerpb.ModelFile) uint64 {
	if h == nil || h.cache == nil {
		return 0
	}
	for _, f := range files {
		if f == nil {
			continue
		}
		if f.GetRole() != workerpb.ModelFileRole_MODEL_FILE_ROLE_PRIMARY {
			continue
		}
		if p := f.GetLocalPath(); p != "" {
			return h.cache.parameterCount(p)
		}
	}
	return 0
}

// primaryThinking reports whether the dispatch's primary model has the
// thinking capability flag set in the catalogue. Used by [predictCost]
// to pick the right decode-token estimate: thinking models generate
// long internal reasoning before answering, so their decode count
// regularly reaches the max_tokens cap and the chatDecodeRatio
// heuristic would under-predict by an order of magnitude.
func (h *handlers) primaryThinking(files []*workerpb.ModelFile) bool {
	if h == nil || h.cache == nil {
		return false
	}
	for _, f := range files {
		if f == nil {
			continue
		}
		if f.GetRole() != workerpb.ModelFileRole_MODEL_FILE_ROLE_PRIMARY {
			continue
		}
		if p := f.GetLocalPath(); p != "" {
			return h.cache.thinking(p)
		}
	}
	return false
}

// visionParamsFor reads the vision-encoder shape (patch size, spatial
// merge) for the dispatch's companion mmproj from the catalogue, so
// the image-token estimate tiles with the model's real geometry
// instead of the package defaults. Zero value when no mmproj rides the
// submit or its entry lacks the metadata.
func (h *handlers) visionParamsFor(files []*workerpb.ModelFile) visionParams {
	if h == nil || h.cache == nil {
		return visionParams{}
	}
	for _, f := range files {
		if f == nil || f.GetRole() != workerpb.ModelFileRole_MODEL_FILE_ROLE_MMPROJ {
			continue
		}
		p := f.GetLocalPath()
		if p == "" {
			continue
		}
		props := h.cache.properties(p)
		vp := visionParams{patchPixels: int(atoiUint64(props["vision_patch_size"]))}
		if m := int(atoiUint64(props["vision_merge_size"])); m > 0 {
			vp.mergeFactor = m * m
		}
		return vp
	}
	return visionParams{}
}

// loadByteEstimate returns the gateway's predictions for the three
// load-cost fields MASS plumbs through Submit:
//
//   - base    : fixed device-memory cost (weights + scratch) the load
//     pays regardless of concurrency.
//   - perSlot : incremental cost per additional context slot (KV at
//     the configured ctx). 0 when GGUF metadata is too sparse to size
//     it honestly — the projection collapses to pool=1.
//   - headroom: the operator's explicit per-load watermark override
//     from LoadHints, or 0 when none was given. The worker applies a
//     per-load hint over its own --vram-headroom-pct flag, so MASS
//     must see the override to project the pool the worker will
//     actually grow; absent an override MASS reads the worker's
//     registration-reported flag instead — sending a gateway-side
//     default here would mask it.
//
// Returns (0, 0, 0) when fileBytes is 0 or no primary path is
// resolvable — MASS treats base=0 as "skip the eligibility check"
// and falls back to pay-on-failure.
func (h *handlers) loadByteEstimate(files []*workerpb.ModelFile, hints *llamacpp.LoadHints) (base, perSlot int64, headroom int32) {
	if h == nil || h.cache == nil {
		return 0, 0, 0
	}
	var fileBytes int64
	var primaryPath string
	for _, f := range files {
		if f == nil {
			continue
		}
		fileBytes += f.GetSizeBytes()
		if f.GetRole() == workerpb.ModelFileRole_MODEL_FILE_ROLE_PRIMARY && primaryPath == "" {
			primaryPath = f.GetLocalPath()
		}
	}
	if fileBytes <= 0 || primaryPath == "" {
		return 0, 0, 0
	}
	props := h.cache.properties(primaryPath)
	base, perSlot = estimateLoadBytes(fileBytes, props, hints.GetContextSize())
	if hints != nil && hints.VramHeadroomPct != nil && *hints.VramHeadroomPct > 0 {
		headroom = *hints.VramHeadroomPct
	}
	return base, perSlot, headroom
}

func (h *handlers) buildChatJob(req *chatRequest, stream bool) (*llamacpp.Job, string, error) {
	storePath, err := h.resolveStorePath(req.Model)
	if err != nil {
		return nil, "", err
	}
	hints, _ := buildLoadHints(req.Config, llamacpp.LoadKind_LOAD_KIND_CHAT, "")
	modelID := model.ID(storePath, hints)
	job := chatJobFromMessages(req.Messages, req.Sampling, stream)
	return job, modelID, nil
}

// buildLoadArtifacts produces the hints + load files for a request. files
// today is just the loopback path of the resolved model (workers on the
// same host load from disk in place). Future remote workers will need URL-
// based ModelFile entries.
func (h *handlers) buildLoadArtifacts(modelStr string, cfg *loadConfig, kind llamacpp.LoadKind) (*llamacpp.LoadHints, []*workerpb.ModelFile, error) {
	storePath, err := h.resolveStorePath(modelStr)
	if err != nil {
		return nil, nil, err
	}
	primaryAbs := absForStoreID(h.modelsDir, storePath)

	// Resolve the vision projector. An explicit cfg.MmprojFilename wins;
	// otherwise, for CHAT loads, auto-attach the sibling projector the
	// gateway knows about (same operator-typed Name). Clients never have
	// to know companion filenames — that's gateway-private store knowledge.
	mmprojAbs := ""
	if cfg != nil && cfg.MmprojFilename != "" {
		// mmproj lives in the same directory as the primary, by convention.
		mmprojAbs = absForStoreID(h.modelsDir,
			filepath.ToSlash(filepath.Join(filepath.Dir(storePath), cfg.MmprojFilename)))
	} else if kind == llamacpp.LoadKind_LOAD_KIND_CHAT && h.cache != nil {
		mmprojAbs = h.cache.companionMmprojPath(primaryAbs)
	}

	mmprojBase := ""
	mmprojStorePath := ""
	if mmprojAbs != "" {
		mmprojBase = filepath.Base(mmprojAbs)
		// The projector lives in the primary's directory by convention.
		mmprojStorePath = path.Join(path.Dir(storePath), mmprojBase)
	}
	hints, _ := buildLoadHints(cfg, kind, mmprojBase)

	// ModelFile.filename is the FULL store-relative cache key: the path
	// relative to the models root WITH the runtime-owned first segment
	// (formatDir, "gguf/…") — the same key MASS holds, DownloadFile
	// carries, and the worker echoes into LoadedModelStatus.files. Base
	// alone would collide across groups and break residency matching.
	// Sha256 lets a remote worker verify integrity on cache reuse and
	// post-download; "" (missing entry or unhashable file) makes it skip
	// verification. Guarded by the same h.cache != nil pattern as the
	// mmproj auto-attach above.
	primarySha := ""
	if h.cache != nil {
		primarySha = h.cache.sha256For(primaryAbs)
	}
	files := []*workerpb.ModelFile{
		{
			Filename:  storeKey(storePath),
			LocalPath: primaryAbs,
			Role:      workerpb.ModelFileRole_MODEL_FILE_ROLE_PRIMARY,
			SizeBytes: fileSize(primaryAbs),
			Sha256:    primarySha,
		},
	}
	if mmprojAbs != "" {
		mmprojSha := ""
		if h.cache != nil {
			mmprojSha = h.cache.sha256For(mmprojAbs)
		}
		files = append(files, &workerpb.ModelFile{
			Filename:  storeKey(mmprojStorePath),
			LocalPath: mmprojAbs,
			Role:      workerpb.ModelFileRole_MODEL_FILE_ROLE_MMPROJ,
			SizeBytes: fileSize(mmprojAbs),
			Sha256:    mmprojSha,
		})
	}
	return hints, files, nil
}

// storeKey turns an under-format-root store path (the request-facing
// model string, e.g. "pdf2doc/x.gguf") into the full store-relative cache
// key by prepending the runtime-owned first segment (formatDir). This is
// the ModelFile.filename / catalogue-key / DownloadFile.rel_path
// namespace — one place so the prefix can never drift.
func storeKey(storePath string) string {
	return formatDir + "/" + storePath
}

// fileSize stats path and returns its size, or 0 on any failure (missing
// file, permission denied, symlink loop). MASS treats a missing size as
// "no penalty" so a stat error doesn't block scheduling.
func fileSize(path string) int64 {
	fi, err := os.Stat(path)
	if err != nil {
		return 0
	}
	return fi.Size()
}

func buildLoadHints(cfg *loadConfig, kind llamacpp.LoadKind, mmprojBase string) (*llamacpp.LoadHints, error) {
	h := &llamacpp.LoadHints{
		Kind:           kind,
		MmprojFilename: mmprojBase,
	}
	if cfg == nil {
		return h, nil
	}
	h.ContextSize = cfg.ContextSize
	h.BatchSize = cfg.BatchSize
	h.GpuLayers = cfg.GPULayers
	h.FlashAttn = cfg.FlashAttn
	h.Threads = cfg.Threads
	h.MaxConcurrent = cfg.MaxConcurrent
	h.VramHeadroomPct = cfg.VramHeadroomPct
	h.Thinking = cfg.Thinking
	h.ChatTemplate = cfg.ChatTemplate
	switch cfg.CacheType {
	case "f16":
		h.CacheType = llamacpp.CacheType_CACHE_TYPE_F16
	case "q8_0":
		h.CacheType = llamacpp.CacheType_CACHE_TYPE_Q8_0
	case "q4_0":
		h.CacheType = llamacpp.CacheType_CACHE_TYPE_Q4_0
	}
	return h, nil
}

func chatMessagesToProto(msgs []chatMessage) []*llamacpp.ChatMessage {
	pbMsgs := make([]*llamacpp.ChatMessage, len(msgs))
	for i, m := range msgs {
		pbMsgs[i] = &llamacpp.ChatMessage{
			Role:    parseRole(m.Role),
			Content: m.Content,
			Parts:   convertParts(m.Parts),
		}
	}
	return pbMsgs
}

func chatJobFromMessages(msgs []chatMessage, s *samplingParams, stream bool) *llamacpp.Job {
	return &llamacpp.Job{
		Kind: llamacpp.JobKind_JOB_KIND_CHAT,
		Body: &llamacpp.Job_Chat{Chat: &llamacpp.ChatJob{
			Messages: chatMessagesToProto(msgs),
			Sampling: convertSampling(s),
			Stream:   stream,
		}},
	}
}

func convertParts(in []contentPart) []*llamacpp.ContentPart {
	out := make([]*llamacpp.ContentPart, 0, len(in))
	for _, p := range in {
		switch p.Type {
		case "text":
			out = append(out, &llamacpp.ContentPart{Content: &llamacpp.ContentPart_Text{Text: p.Text}})
		case "image":
			out = append(out, &llamacpp.ContentPart{Content: &llamacpp.ContentPart_Image{Image: &llamacpp.ImageContent{Data: p.Data, MimeType: p.MIMEType}}})
		case "audio":
			out = append(out, &llamacpp.ContentPart{Content: &llamacpp.ContentPart_Audio{Audio: &llamacpp.AudioContent{Data: p.Data, MimeType: p.MIMEType}}})
		}
	}
	return out
}

func convertSampling(s *samplingParams) *llamacpp.SamplingParams {
	if s == nil {
		return nil
	}
	return &llamacpp.SamplingParams{
		MaxTokens:        s.MaxTokens,
		Temperature:      s.Temperature,
		TopP:             s.TopP,
		TopK:             s.TopK,
		Seed:             s.Seed,
		Stop:             s.Stop,
		MinP:             s.MinP,
		RepeatPenalty:    s.RepeatPenalty,
		FrequencyPenalty: s.FrequencyPenalty,
		PresencePenalty:  s.PresencePenalty,
		EnableThinking:   s.EnableThinking,
	}
}

func parseRole(r string) llamacpp.Role {
	switch r {
	case "system":
		return llamacpp.Role_ROLE_SYSTEM
	case "user":
		return llamacpp.Role_ROLE_USER
	case "assistant":
		return llamacpp.Role_ROLE_ASSISTANT
	case "tool":
		return llamacpp.Role_ROLE_TOOL
	default:
		return llamacpp.Role_ROLE_UNSPECIFIED
	}
}

func roleString(r llamacpp.Role) string {
	switch r {
	case llamacpp.Role_ROLE_SYSTEM:
		return "system"
	case llamacpp.Role_ROLE_USER:
		return "user"
	case llamacpp.Role_ROLE_ASSISTANT:
		return "assistant"
	case llamacpp.Role_ROLE_TOOL:
		return "tool"
	default:
		return ""
	}
}

func finishReasonString(f llamacpp.FinishReason) string {
	switch f {
	case llamacpp.FinishReason_FINISH_REASON_STOP:
		return "stop"
	case llamacpp.FinishReason_FINISH_REASON_LENGTH:
		return "length"
	case llamacpp.FinishReason_FINISH_REASON_CONTENT_FILTER:
		return "content_filter"
	case llamacpp.FinishReason_FINISH_REASON_TOOL_CALLS:
		return "tool_calls"
	default:
		return ""
	}
}

func applyChatFinal(resp *chatResponse, cf *llamacpp.ChatFinal) {
	if cf.GetMessage() != nil {
		m := cf.GetMessage()
		resp.Message = &chatMessage{Role: roleString(m.GetRole()), Content: m.GetContent()}
	}
	resp.FinishReason = finishReasonString(cf.GetFinishReason())
	resp.ReasoningContent = cf.GetReasoningContent()
	if cf.GetUsage() != nil {
		u := cf.GetUsage()
		resp.Usage = &usage{PromptTokens: u.GetPromptTokens(), CompletionTokens: u.GetCompletionTokens(), TotalTokens: u.GetTotalTokens()}
	}
	resp.TokensPerSecond = cf.GetTokensPerSecond()
}

// ----- IO helpers -----

func readJSON(r *http.Request, dst any) error {
	body, err := io.ReadAll(r.Body)
	if err != nil {
		return ctxerr.With(fmt.Errorf("reading request body: %w", err), nil)
	}
	if err := json.Unmarshal(body, dst); err != nil {
		return ctxerr.With(fmt.Errorf("parsing request: %w", err), nil)
	}
	return nil
}

func writeJSON(w http.ResponseWriter, status int, body any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(body); err != nil && !errors.Is(err, http.ErrHandlerTimeout) {
		// Best-effort: nothing useful to do on a closed client.
		_ = err
	}
}

func writeError(w http.ResponseWriter, status int, err error) {
	writeJSON(w, status, map[string]string{"error": err.Error()})
}

// statusError wraps an error with the HTTP status a handler should return.
// Helpers that build a chunk stream return these so the calling handler maps
// to the right code without re-deciding (build/load = 400, dispatch = 500).
type statusError struct {
	status int
	err    error
}

func (e *statusError) Error() string { return e.err.Error() }
func (e *statusError) Unwrap() error { return e.err }

func badRequest(err error) error { return &statusError{http.StatusBadRequest, err} }

// writeStatusError writes the status carried by a *statusError, defaulting to
// 500 for a plain error.
func writeStatusError(w http.ResponseWriter, err error) {
	var se *statusError
	if errors.As(err, &se) {
		writeError(w, se.status, se.err)
		return
	}
	writeError(w, http.StatusInternalServerError, err)
}

func writeSSE(w io.Writer, body string) {
	_, _ = io.WriteString(w, "data: "+body+"\n\n")
}
