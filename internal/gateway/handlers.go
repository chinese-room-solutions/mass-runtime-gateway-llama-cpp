package gateway

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"github.com/KernelPryanic/ctxerr"
	llamacpp "github.com/chinese-room-solutions/mass-proto/gen/go/llama-cpp"
	workerpb "github.com/chinese-room-solutions/mass-proto/gen/go/worker"
	"github.com/chinese-room-solutions/mass-runtime-llama-cpp/internal/huggingface"
	"github.com/chinese-room-solutions/mass-runtime-llama-cpp/internal/model"
	"github.com/chinese-room-solutions/mass-runtime-llama-cpp/internal/payload"
	"github.com/chinese-room-solutions/mass-runtime-llama-cpp/internal/sched"
	"github.com/google/uuid"
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
	Model    string         `json:"model"`
	Config   *loadConfig    `json:"config,omitempty"`   // optional; per-request load overrides
	Messages []chatMessage  `json:"messages"`
	Sampling *samplingParams `json:"sampling,omitempty"`
}

type batchChatRequest struct {
	Model  string             `json:"model"`
	Config *loadConfig        `json:"config,omitempty"`
	Items  []batchChatItem    `json:"items"`
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

type loadModelRequest struct {
	Model  string      `json:"model"`
	Config *loadConfig `json:"config,omitempty"`
	Kind   string      `json:"kind,omitempty"` // "chat" | "embedding"; default "chat"
}

type downloadRequest struct {
	RepoID   string `json:"repo_id"`
	Filename string `json:"filename"`
}

// loadConfig mirrors llamacpp.LoadHints in JSON form. Pointer fields use
// optional-style JSON ("set" iff present in the body).
type loadConfig struct {
	ContextSize    int32    `json:"context_size,omitempty"`
	BatchSize      *int32   `json:"batch_size,omitempty"`
	GPULayers      *int32   `json:"gpu_layers,omitempty"`
	FlashAttn      *bool    `json:"flash_attn,omitempty"`
	Threads        *int32   `json:"threads,omitempty"`
	MaxConcurrent  *int32   `json:"max_concurrent,omitempty"`
	Thinking       bool     `json:"thinking,omitempty"`
	MainGPU        string   `json:"main_gpu,omitempty"`
	TensorSplit    []float32 `json:"tensor_split,omitempty"`
	MmprojFilename string   `json:"mmproj_filename,omitempty"`
	ChatTemplate   string   `json:"chat_template,omitempty"`
	CacheType      string   `json:"cache_type,omitempty"` // "f16" | "q8_0" | "q4_0"
}

type chatMessage struct {
	Role    string        `json:"role"`
	Content string        `json:"content,omitempty"`
	Parts   []contentPart `json:"parts,omitempty"`
}

type contentPart struct {
	Type     string  `json:"type"` // "text" | "image" | "audio"
	Text     string  `json:"text,omitempty"`
	Data     []byte  `json:"data,omitempty"`
	MIMEType string  `json:"mime_type,omitempty"`
}

type samplingParams struct {
	MaxTokens        *int32   `json:"max_tokens,omitempty"`
	Temperature      float32  `json:"temperature,omitempty"`
	TopP             float32  `json:"top_p,omitempty"`
	TopK             int32    `json:"top_k,omitempty"`
	Seed             *int32   `json:"seed,omitempty"`
	Stop             []string `json:"stop,omitempty"`
	MinP             float32  `json:"min_p,omitempty"`
	RepeatPenalty    float32  `json:"repeat_penalty,omitempty"`
	FrequencyPenalty float32  `json:"frequency_penalty,omitempty"`
	PresencePenalty  float32  `json:"presence_penalty,omitempty"`
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
	ID         string           `json:"id"`
	Model      string           `json:"model"`
	Embeddings []embeddingItem  `json:"embeddings"`
}

type embeddingItem struct {
	Index     int32     `json:"index"`
	Embedding []float32 `json:"embedding"`
}

type tokenizeResponse struct {
	Tokens []int32 `json:"tokens"`
}

type loadModelResponse struct {
	ModelID  string `json:"model_id"`
	WorkerID string `json:"worker_id"`
	PoolSize int32  `json:"pool_size"`
}

// ----- Handlers -----

func (h *handlers) handleChat(w http.ResponseWriter, r *http.Request) {
	var req chatRequest
	if err := readJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	job, modelID, err := h.buildChatJob(&req, false)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	hints, files, err := h.buildLoadArtifacts(req.Model, req.Config, llamacpp.LoadKind_LOAD_KIND_CHAT)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}

	chunks, err := h.dispatch(r.Context(), modelID, job, hints, files)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}

	resp := chatResponse{ID: uuid.NewString(), Model: req.Model}
	for c := range chunks {
		switch c.Type {
		case sched.ChunkBody:
			// Sync chat shouldn't see body chunks, but if the worker emits
			// them just append to the message.
			if dec, decErr := payload.DecodeJobChunk(c.Chunk); decErr == nil {
				if cf := dec.GetChatFinal(); cf != nil {
					applyChatFinal(&resp, cf)
				}
			}
		case sched.ChunkCompleted:
			if len(c.Final) > 0 {
				if dec, decErr := payload.DecodeJobChunk(c.Final); decErr == nil {
					if cf := dec.GetChatFinal(); cf != nil {
						applyChatFinal(&resp, cf)
					}
				}
			}
		case sched.ChunkError:
			writeError(w, http.StatusInternalServerError, fmt.Errorf("worker: %s", c.ErrText))
			return
		}
	}
	writeJSON(w, http.StatusOK, resp)
}

func (h *handlers) handleChatStream(w http.ResponseWriter, r *http.Request) {
	var req chatRequest
	if err := readJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	job, modelID, err := h.buildChatJob(&req, true)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	hints, files, err := h.buildLoadArtifacts(req.Model, req.Config, llamacpp.LoadKind_LOAD_KIND_CHAT)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}

	chunks, err := h.dispatch(r.Context(), modelID, job, hints, files)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	flusher, _ := w.(http.Flusher)
	id := uuid.NewString()
	for c := range chunks {
		switch c.Type {
		case sched.ChunkBody:
			dec, decErr := payload.DecodeJobChunk(c.Chunk)
			if decErr != nil {
				continue
			}
			if delta := dec.GetChat(); delta != nil {
				writeSSE(w, sseChatFrame(id, req.Model, delta, ""))
			}
		case sched.ChunkCompleted:
			if len(c.Final) > 0 {
				if dec, decErr := payload.DecodeJobChunk(c.Final); decErr == nil {
					if cf := dec.GetChatFinal(); cf != nil {
						writeSSE(w, sseChatFinalFrame(id, req.Model, cf))
					}
				}
			}
			writeSSE(w, "[DONE]")
		case sched.ChunkError:
			writeSSE(w, fmt.Sprintf(`{"error":%q}`, c.ErrText))
			return
		}
		if flusher != nil {
			flusher.Flush()
		}
	}
}

func (h *handlers) handleBatchChat(w http.ResponseWriter, r *http.Request) {
	var req batchChatRequest
	if err := readJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	if len(req.Items) == 0 {
		writeError(w, http.StatusBadRequest, fmt.Errorf("items must be non-empty"))
		return
	}
	hints, files, err := h.buildLoadArtifacts(req.Model, req.Config, llamacpp.LoadKind_LOAD_KIND_CHAT)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	storePath := model.ResolveModelPath(req.Model)
	if storePath == "" {
		writeError(w, http.StatusBadRequest, fmt.Errorf("model: invalid path %q", req.Model))
		return
	}
	modelID := model.ID(storePath, hints)

	// Pre-load once for the whole batch — the scheduler short-circuits when
	// the model is already loaded, so per-item dispatches are pure schedule
	// calls without re-shipping files.
	if err := h.ensureLoaded(r.Context(), modelID, hints, files); err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}

	out := make([]chatResponse, len(req.Items))
	for i, item := range req.Items {
		j := chatJobFromMessages(item.Messages, item.Sampling, false)
		bytesJob, err := payload.EncodeJob(j)
		if err != nil {
			writeError(w, http.StatusInternalServerError, err)
			return
		}
		chunks, err := h.scheduler.Schedule(r.Context(), sched.ScheduleParams{
			ModelID: modelID,
			Payload: bytesJob,
		})
		if err != nil {
			writeError(w, http.StatusInternalServerError, err)
			return
		}
		// Drain (single-shot, sync chat).
		out[i] = chatResponse{ID: uuid.NewString(), Model: req.Model}
		for c := range chunks {
			if c.Type == sched.ChunkCompleted && len(c.Final) > 0 {
				if dec, decErr := payload.DecodeJobChunk(c.Final); decErr == nil {
					if cf := dec.GetChatFinal(); cf != nil {
						applyChatFinal(&out[i], cf)
					}
				}
			}
			if c.Type == sched.ChunkError {
				writeError(w, http.StatusInternalServerError, fmt.Errorf("item %d: %s", i, c.ErrText))
				return
			}
		}
	}
	writeJSON(w, http.StatusOK, struct {
		Responses []chatResponse `json:"responses"`
	}{Responses: out})
}

func (h *handlers) handleEmbed(w http.ResponseWriter, r *http.Request) {
	var req embedRequest
	if err := readJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	job := &llamacpp.Job{
		Kind: llamacpp.JobKind_JOB_KIND_EMBED,
		Body: &llamacpp.Job_Embed{Embed: &llamacpp.EmbedJob{Input: req.Input}},
	}
	hints, files, err := h.buildLoadArtifacts(req.Model, req.Config, llamacpp.LoadKind_LOAD_KIND_EMBEDDING)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	storePath := model.ResolveModelPath(req.Model)
	if storePath == "" {
		writeError(w, http.StatusBadRequest, fmt.Errorf("model: invalid path %q", req.Model))
		return
	}
	modelID := model.ID(storePath, hints)
	chunks, err := h.dispatch(r.Context(), modelID, job, hints, files)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	resp := embedResponse{ID: uuid.NewString(), Model: req.Model}
	for c := range chunks {
		if c.Type == sched.ChunkCompleted && len(c.Final) > 0 {
			dec, decErr := payload.DecodeJobChunk(c.Final)
			if decErr == nil {
				if er := dec.GetEmbed(); er != nil {
					resp.Embedding = er.GetEmbedding()
				}
			}
		}
		if c.Type == sched.ChunkError {
			writeError(w, http.StatusInternalServerError, fmt.Errorf("worker: %s", c.ErrText))
			return
		}
	}
	writeJSON(w, http.StatusOK, resp)
}

func (h *handlers) handleBatchEmbed(w http.ResponseWriter, r *http.Request) {
	var req batchEmbedRequest
	if err := readJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	job := &llamacpp.Job{
		Kind: llamacpp.JobKind_JOB_KIND_BATCH_EMBED,
		Body: &llamacpp.Job_BatchEmbed{BatchEmbed: &llamacpp.BatchEmbedJob{Inputs: req.Inputs}},
	}
	hints, files, err := h.buildLoadArtifacts(req.Model, req.Config, llamacpp.LoadKind_LOAD_KIND_EMBEDDING)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	storePath := model.ResolveModelPath(req.Model)
	if storePath == "" {
		writeError(w, http.StatusBadRequest, fmt.Errorf("model: invalid path %q", req.Model))
		return
	}
	modelID := model.ID(storePath, hints)
	chunks, err := h.dispatch(r.Context(), modelID, job, hints, files)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	resp := batchEmbedResponse{ID: uuid.NewString(), Model: req.Model}
	for c := range chunks {
		if c.Type == sched.ChunkCompleted && len(c.Final) > 0 {
			if dec, decErr := payload.DecodeJobChunk(c.Final); decErr == nil {
				if br := dec.GetBatchEmbed(); br != nil {
					resp.Embeddings = make([]embeddingItem, len(br.GetItems()))
					for i, it := range br.GetItems() {
						resp.Embeddings[i] = embeddingItem{Index: it.GetIndex(), Embedding: it.GetEmbedding()}
					}
				}
			}
		}
		if c.Type == sched.ChunkError {
			writeError(w, http.StatusInternalServerError, fmt.Errorf("worker: %s", c.ErrText))
			return
		}
	}
	writeJSON(w, http.StatusOK, resp)
}

func (h *handlers) handleTokenize(w http.ResponseWriter, r *http.Request) {
	var req tokenizeRequest
	if err := readJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	job := &llamacpp.Job{
		Kind: llamacpp.JobKind_JOB_KIND_TOKENIZE,
		Body: &llamacpp.Job_Tokenize{Tokenize: &llamacpp.TokenizeJob{Text: req.Text}},
	}
	// Tokenize uses the chat tokenizer in this gateway.
	hints, files, err := h.buildLoadArtifacts(req.Model, req.Config, llamacpp.LoadKind_LOAD_KIND_CHAT)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	storePath := model.ResolveModelPath(req.Model)
	if storePath == "" {
		writeError(w, http.StatusBadRequest, fmt.Errorf("model: invalid path %q", req.Model))
		return
	}
	modelID := model.ID(storePath, hints)
	chunks, err := h.dispatch(r.Context(), modelID, job, hints, files)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	resp := tokenizeResponse{}
	for c := range chunks {
		if c.Type == sched.ChunkCompleted && len(c.Final) > 0 {
			if dec, decErr := payload.DecodeJobChunk(c.Final); decErr == nil {
				if tr := dec.GetTokenize(); tr != nil {
					resp.Tokens = tr.GetTokens()
				}
			}
		}
		if c.Type == sched.ChunkError {
			writeError(w, http.StatusInternalServerError, fmt.Errorf("worker: %s", c.ErrText))
			return
		}
	}
	writeJSON(w, http.StatusOK, resp)
}

func (h *handlers) handleLoadModel(w http.ResponseWriter, r *http.Request) {
	var req loadModelRequest
	if err := readJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	kind := llamacpp.LoadKind_LOAD_KIND_CHAT
	if req.Kind == "embedding" {
		kind = llamacpp.LoadKind_LOAD_KIND_EMBEDDING
	}
	hints, files, err := h.buildLoadArtifacts(req.Model, req.Config, kind)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	storePath := model.ResolveModelPath(req.Model)
	if storePath == "" {
		writeError(w, http.StatusBadRequest, fmt.Errorf("model: invalid path %q", req.Model))
		return
	}
	modelID := model.ID(storePath, hints)
	hintsBytes, err := payload.EncodeLoadHints(hints)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	instances, err := h.scheduler.EnsureModelLoaded(r.Context(), sched.EnsureModelLoadedParams{
		ModelID:   modelID,
		Files:     files,
		LoadHints: hintsBytes,
		Source:    sourceFromContext(r.Context()),
	})
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	if len(instances) == 0 {
		writeError(w, http.StatusInternalServerError, fmt.Errorf("scheduler returned no instances"))
		return
	}
	writeJSON(w, http.StatusOK, loadModelResponse{
		ModelID:  modelID,
		WorkerID: instances[0].WorkerID,
		PoolSize: instances[0].PoolSize,
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
	abs := absForStoreID(h.modelsDir, id)
	info, ok := h.cache.parseModelInfo(abs, id)
	if !ok {
		return
	}
	if _, err := io.WriteString(w, renderModelDetail(h.runtimeName, info)); err != nil {
		h.logger.Debug().Err(err).Msg("writing model detail html")
	}
}

// handleDeleteModel removes one model file (and its mmproj companion when
// present) from this runtime's modelsDir. Path is the URL-decoded
// store-relative ID — the same opaque value emitted in [renderModelsList].
//
// We refuse paths that escape modelsDir (defence in depth: filepath.Clean
// alone won't catch a leading "/" or a Windows-style "C:\..." submitted by
// a misbehaving caller).
func (h *handlers) handleDeleteModel(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	if id == "" {
		http.Error(w, "missing model id", http.StatusBadRequest)
		return
	}
	cleaned := filepath.Clean(filepath.FromSlash(id))
	if strings.HasPrefix(cleaned, "..") || strings.HasPrefix(cleaned, string(filepath.Separator)) || filepath.IsAbs(cleaned) {
		http.Error(w, "invalid model id", http.StatusBadRequest)
		return
	}
	root := formatRoot(h.modelsDir)
	abs := filepath.Join(root, cleaned)
	rel, err := filepath.Rel(root, abs)
	if err != nil || strings.HasPrefix(rel, "..") {
		http.Error(w, "invalid model id", http.StatusBadRequest)
		return
	}
	if err := os.Remove(abs); err != nil && !os.IsNotExist(err) {
		http.Error(w, "delete model: "+err.Error(), http.StatusInternalServerError)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

func (h *handlers) handleDownload(w http.ResponseWriter, r *http.Request) {
	var req downloadRequest
	if err := readJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	if req.RepoID == "" || req.Filename == "" {
		writeError(w, http.StatusBadRequest, fmt.Errorf("repo_id and filename required"))
		return
	}
	// Stream progress as SSE.
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	flusher, _ := w.(http.Flusher)

	progress := func(downloaded, total int64) {
		writeSSE(w, fmt.Sprintf(`{"downloaded":%d,"total":%d}`, downloaded, total))
		if flusher != nil {
			flusher.Flush()
		}
	}
	dest, err := huggingface.Download(r.Context(), req.RepoID, req.Filename, h.modelsDir, progress)
	if err != nil {
		writeSSE(w, fmt.Sprintf(`{"error":%q}`, err.Error()))
		return
	}
	relID := huggingface.StoreRelativePath(req.RepoID, req.Filename)
	writeSSE(w, fmt.Sprintf(`{"completed":true,"id":%q,"path":%q}`, relID, dest))
}

// ----- Shared helpers -----

// dispatch is the common path for inference: ensure the model is loaded on
// some worker (passing the file artifacts MASS may need to ship), then
// submit the job. Used by chat (sync + stream), embed, batch-embed, and
// tokenize.
//
// EnsureModelLoaded is idempotent against modelID — if a worker already has
// it loaded, the call short-circuits. So calling per-request is cheap on the
// hot path.
func (h *handlers) dispatch(ctx context.Context, modelID string, job *llamacpp.Job, hints *llamacpp.LoadHints, files []*workerpb.ModelFile) (<-chan sched.JobChunk, error) {
	bodyBytes, err := payload.EncodeJob(job)
	if err != nil {
		return nil, err
	}
	if err := h.ensureLoaded(ctx, modelID, hints, files); err != nil {
		return nil, err
	}
	return h.scheduler.Schedule(ctx, sched.ScheduleParams{
		ModelID: modelID,
		Payload: bodyBytes,
	})
}

// ensureLoaded asks MASS to load modelID with the given files + hints if no
// worker has it resident yet. Wraps the scheduler call in ctxerr so callers
// get a uniform error shape. Source + kind metadata are taken from the
// request context (set by the inbound HTTP / gRPC handler) so MASS can
// attribute the load in its Scheduler tab.
func (h *handlers) ensureLoaded(ctx context.Context, modelID string, hints *llamacpp.LoadHints, files []*workerpb.ModelFile) error {
	hintsBytes, err := payload.EncodeLoadHints(hints)
	if err != nil {
		return ctxerr.With(fmt.Errorf("encoding load hints: %w", err), map[string]any{"model_id": modelID})
	}
	_, err = h.scheduler.EnsureModelLoaded(ctx, sched.EnsureModelLoadedParams{
		ModelID:   modelID,
		Files:     files,
		LoadHints: hintsBytes,
		Source:    sourceFromContext(ctx),
	})
	if err != nil {
		return ctxerr.With(fmt.Errorf("ensuring model loaded: %w", err), map[string]any{"model_id": modelID})
	}
	return nil
}


func (h *handlers) buildChatJob(req *chatRequest, stream bool) (*llamacpp.Job, string, error) {
	storePath := model.ResolveModelPath(req.Model)
	if storePath == "" {
		return nil, "", fmt.Errorf("model: invalid path %q", req.Model)
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
	storePath := model.ResolveModelPath(modelStr)
	if storePath == "" {
		return nil, nil, fmt.Errorf("model: invalid path %q", modelStr)
	}
	mmprojPath := ""
	if cfg != nil && cfg.MmprojFilename != "" {
		// mmproj lives in the same directory as the primary, by convention.
		mmprojPath = filepath.ToSlash(filepath.Join(filepath.Dir(storePath), cfg.MmprojFilename))
	}
	hints, _ := buildLoadHints(cfg, kind, cfg.mmprojFilenameOnly())

	files := []*workerpb.ModelFile{
		{
			Filename:  filepath.Base(storePath),
			LocalPath: absForStoreID(h.modelsDir, storePath),
			Role:      workerpb.ModelFileRole_MODEL_FILE_ROLE_PRIMARY,
		},
	}
	if mmprojPath != "" {
		files = append(files, &workerpb.ModelFile{
			Filename:  filepath.Base(mmprojPath),
			LocalPath: absForStoreID(h.modelsDir, mmprojPath),
			Role:      workerpb.ModelFileRole_MODEL_FILE_ROLE_MMPROJ,
		})
	}
	return hints, files, nil
}

// (mmprojFilenameOnly returns the basename, used in LoadHints)
func (c *loadConfig) mmprojFilenameOnly() string {
	if c == nil {
		return ""
	}
	return filepath.Base(c.MmprojFilename)
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
	h.Thinking = cfg.Thinking
	h.MainGpu = cfg.MainGPU
	h.TensorSplit = cfg.TensorSplit
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

func chatJobFromMessages(msgs []chatMessage, s *samplingParams, stream bool) *llamacpp.Job {
	pbMsgs := make([]*llamacpp.ChatMessage, len(msgs))
	for i, m := range msgs {
		pbMsgs[i] = &llamacpp.ChatMessage{
			Role:    parseRole(m.Role),
			Content: m.Content,
			Parts:   convertParts(m.Parts),
		}
	}
	return &llamacpp.Job{
		Kind: llamacpp.JobKind_JOB_KIND_CHAT,
		Body: &llamacpp.Job_Chat{Chat: &llamacpp.ChatJob{
			Messages: pbMsgs,
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

func writeSSE(w io.Writer, body string) {
	_, _ = io.WriteString(w, "data: "+body+"\n\n")
}

