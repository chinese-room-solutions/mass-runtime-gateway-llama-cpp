package gateway

import (
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	llamacpp "github.com/chinese-room-solutions/mass-proto/gen/go/llama-cpp"
	"github.com/chinese-room-solutions/mass-runtime-llama-cpp/internal/model"
	"github.com/chinese-room-solutions/mass-runtime-llama-cpp/internal/payload"
	"github.com/chinese-room-solutions/mass-runtime-llama-cpp/internal/sched"
	"github.com/google/uuid"
)

// ----- OpenAI request shapes -----

type openAIChatRequest struct {
	Model    string             `json:"model"`
	Messages []openAIChatMessage `json:"messages"`
	Stream   bool               `json:"stream,omitempty"`

	Temperature      float32  `json:"temperature,omitempty"`
	TopP             float32  `json:"top_p,omitempty"`
	MaxTokens        *int32   `json:"max_tokens,omitempty"`
	Stop             []string `json:"stop,omitempty"`
	Seed             *int32   `json:"seed,omitempty"`
	FrequencyPenalty float32  `json:"frequency_penalty,omitempty"`
	PresencePenalty  float32  `json:"presence_penalty,omitempty"`
}

type openAIChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ----- OpenAI response shapes -----

type openAIChatResponse struct {
	ID      string         `json:"id"`
	Object  string         `json:"object"`
	Created int64          `json:"created"`
	Model   string         `json:"model"`
	Choices []openAIChoice `json:"choices"`
	Usage   *openAIUsage   `json:"usage,omitempty"`
}

type openAIChoice struct {
	Index        int                `json:"index"`
	Message      *openAIChatMessage `json:"message,omitempty"`
	Delta        *openAIChatMessage `json:"delta,omitempty"`
	FinishReason string             `json:"finish_reason,omitempty"`
}

type openAIUsage struct {
	PromptTokens     int32 `json:"prompt_tokens"`
	CompletionTokens int32 `json:"completion_tokens"`
	TotalTokens      int32 `json:"total_tokens"`
}

type openAIEmbedRequest struct {
	Model string `json:"model"`
	Input any    `json:"input"` // string | []string
}

type openAIEmbedResponse struct {
	Object string             `json:"object"`
	Data   []openAIEmbedItem  `json:"data"`
	Model  string             `json:"model"`
	Usage  *openAIUsage       `json:"usage,omitempty"`
}

type openAIEmbedItem struct {
	Object    string    `json:"object"`
	Index     int       `json:"index"`
	Embedding []float32 `json:"embedding"`
}

type openAIModelsResponse struct {
	Object string           `json:"object"`
	Data   []openAIModelRef `json:"data"`
}

type openAIModelRef struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	OwnedBy string `json:"owned_by"`
}

// ----- Handlers -----

func (h *handlers) handleOpenAIChat(w http.ResponseWriter, r *http.Request) {
	var req openAIChatRequest
	if err := readJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	storePath := model.ResolveModelPath(req.Model)
	if storePath == "" {
		writeError(w, http.StatusBadRequest, fmt.Errorf("model: invalid path %q", req.Model))
		return
	}
	hints, files, err := h.buildLoadArtifacts(req.Model, nil, llamacpp.LoadKind_LOAD_KIND_CHAT)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	modelID := model.ID(storePath, hints)

	pbMsgs := make([]*llamacpp.ChatMessage, len(req.Messages))
	for i, m := range req.Messages {
		pbMsgs[i] = &llamacpp.ChatMessage{Role: parseRole(m.Role), Content: m.Content}
	}

	job := &llamacpp.Job{
		Kind: llamacpp.JobKind_JOB_KIND_CHAT,
		Body: &llamacpp.Job_Chat{Chat: &llamacpp.ChatJob{
			Messages: pbMsgs,
			Sampling: &llamacpp.SamplingParams{
				MaxTokens:        req.MaxTokens,
				Temperature:      req.Temperature,
				TopP:             req.TopP,
				Stop:             req.Stop,
				Seed:             req.Seed,
				FrequencyPenalty: req.FrequencyPenalty,
				PresencePenalty:  req.PresencePenalty,
			},
			Stream: req.Stream,
		}},
	}
	chunks, err := h.dispatch(r.Context(), modelID, job, hints, files)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}

	id := "chatcmpl-" + uuid.NewString()
	created := time.Now().Unix()

	if req.Stream {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		flusher, _ := w.(http.Flusher)
		for c := range chunks {
			switch c.Type {
			case sched.ChunkBody:
				if dec, derr := payload.DecodeJobChunk(c.Chunk); derr == nil {
					if delta := dec.GetChat(); delta != nil {
						frame := openAIChatResponse{
							ID: id, Object: "chat.completion.chunk", Created: created, Model: req.Model,
							Choices: []openAIChoice{{
								Index: 0,
								Delta: &openAIChatMessage{Role: roleString(delta.GetRole()), Content: delta.GetContent()},
							}},
						}
						b, _ := json.Marshal(frame)
						writeSSE(w, string(b))
					}
				}
			case sched.ChunkCompleted:
				if len(c.Final) > 0 {
					if dec, derr := payload.DecodeJobChunk(c.Final); derr == nil {
						if cf := dec.GetChatFinal(); cf != nil {
							frame := openAIChatResponse{
								ID: id, Object: "chat.completion.chunk", Created: created, Model: req.Model,
								Choices: []openAIChoice{{
									Index:        0,
									Delta:        &openAIChatMessage{},
									FinishReason: finishReasonString(cf.GetFinishReason()),
								}},
							}
							b, _ := json.Marshal(frame)
							writeSSE(w, string(b))
						}
					}
				}
				writeSSE(w, "[DONE]")
			case sched.ChunkError:
				writeSSE(w, fmt.Sprintf(`{"error":{"message":%q}}`, c.ErrText))
				return
			}
			if flusher != nil {
				flusher.Flush()
			}
		}
		return
	}

	// Non-streaming: assemble a single response.
	resp := openAIChatResponse{
		ID: id, Object: "chat.completion", Created: created, Model: req.Model,
		Choices: []openAIChoice{{Index: 0}},
	}
	for c := range chunks {
		if c.Type == sched.ChunkCompleted && len(c.Final) > 0 {
			if dec, derr := payload.DecodeJobChunk(c.Final); derr == nil {
				if cf := dec.GetChatFinal(); cf != nil {
					if cf.GetMessage() != nil {
						resp.Choices[0].Message = &openAIChatMessage{
							Role:    roleString(cf.GetMessage().GetRole()),
							Content: cf.GetMessage().GetContent(),
						}
					}
					resp.Choices[0].FinishReason = finishReasonString(cf.GetFinishReason())
					if cf.GetUsage() != nil {
						u := cf.GetUsage()
						resp.Usage = &openAIUsage{
							PromptTokens: u.GetPromptTokens(), CompletionTokens: u.GetCompletionTokens(), TotalTokens: u.GetTotalTokens(),
						}
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

func (h *handlers) handleOpenAIEmbed(w http.ResponseWriter, r *http.Request) {
	var req openAIEmbedRequest
	if err := readJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	storePath := model.ResolveModelPath(req.Model)
	if storePath == "" {
		writeError(w, http.StatusBadRequest, fmt.Errorf("model: invalid path %q", req.Model))
		return
	}
	hints, files, err := h.buildLoadArtifacts(req.Model, nil, llamacpp.LoadKind_LOAD_KIND_EMBEDDING)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	modelID := model.ID(storePath, hints)

	inputs := normaliseEmbedInput(req.Input)
	if len(inputs) == 0 {
		writeError(w, http.StatusBadRequest, fmt.Errorf("input is required"))
		return
	}

	var job *llamacpp.Job
	if len(inputs) == 1 {
		job = &llamacpp.Job{
			Kind: llamacpp.JobKind_JOB_KIND_EMBED,
			Body: &llamacpp.Job_Embed{Embed: &llamacpp.EmbedJob{Input: inputs[0]}},
		}
	} else {
		job = &llamacpp.Job{
			Kind: llamacpp.JobKind_JOB_KIND_BATCH_EMBED,
			Body: &llamacpp.Job_BatchEmbed{BatchEmbed: &llamacpp.BatchEmbedJob{Inputs: inputs}},
		}
	}
	chunks, err := h.dispatch(r.Context(), modelID, job, hints, files)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}

	resp := openAIEmbedResponse{Object: "list", Model: req.Model}
	for c := range chunks {
		if c.Type == sched.ChunkCompleted && len(c.Final) > 0 {
			if dec, derr := payload.DecodeJobChunk(c.Final); derr == nil {
				if er := dec.GetEmbed(); er != nil {
					resp.Data = []openAIEmbedItem{{Object: "embedding", Index: 0, Embedding: er.GetEmbedding()}}
				}
				if br := dec.GetBatchEmbed(); br != nil {
					resp.Data = make([]openAIEmbedItem, len(br.GetItems()))
					for i, it := range br.GetItems() {
						resp.Data[i] = openAIEmbedItem{Object: "embedding", Index: int(it.GetIndex()), Embedding: it.GetEmbedding()}
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

func (h *handlers) handleOpenAIModels(w http.ResponseWriter, r *http.Request) {
	_ = r
	infos, err := h.cache.walkAndParseModels(formatRoot(h.modelsDir))
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	data := make([]openAIModelRef, 0, len(infos))
	for _, info := range infos {
		// /v1/models hides companion files — apps never select them directly.
		if info.Companion != "" {
			continue
		}
		data = append(data, openAIModelRef{ID: info.ID, Object: "model", OwnedBy: "llama-cpp"})
	}
	writeJSON(w, http.StatusOK, openAIModelsResponse{Object: "list", Data: data})
}

// normaliseEmbedInput accepts either a single string or an array of strings
// (the OpenAI v1/embeddings spec).
func normaliseEmbedInput(in any) []string {
	switch v := in.(type) {
	case string:
		if v == "" {
			return nil
		}
		return []string{v}
	case []any:
		out := make([]string, 0, len(v))
		for _, x := range v {
			if s, ok := x.(string); ok && s != "" {
				out = append(out, s)
			}
		}
		return out
	}
	return nil
}

// ----- SSE chat frame helpers (typed-API streaming) -----

func sseChatFrame(id, modelStr string, delta *llamacpp.ChatChunk, _ string) string {
	frame := chatStreamFrame{
		ID:    id,
		Model: modelStr,
		Delta: chatStreamDelta{
			Role:             roleString(delta.GetRole()),
			Content:          delta.GetContent(),
			ReasoningContent: delta.GetReasoningContent(),
		},
	}
	b, _ := json.Marshal(frame)
	return string(b)
}

func sseChatFinalFrame(id, modelStr string, cf *llamacpp.ChatFinal) string {
	frame := chatStreamFrame{
		ID:           id,
		Model:        modelStr,
		FinishReason: finishReasonString(cf.GetFinishReason()),
	}
	if cf.GetUsage() != nil {
		u := cf.GetUsage()
		frame.Usage = &usage{PromptTokens: u.GetPromptTokens(), CompletionTokens: u.GetCompletionTokens(), TotalTokens: u.GetTotalTokens()}
	}
	frame.TokensPerSecond = cf.GetTokensPerSecond()
	b, _ := json.Marshal(frame)
	return string(b)
}

type chatStreamFrame struct {
	ID              string          `json:"id"`
	Model           string          `json:"model"`
	Delta           chatStreamDelta `json:"delta,omitzero"`
	FinishReason    string          `json:"finish_reason,omitempty"`
	Usage           *usage          `json:"usage,omitempty"`
	TokensPerSecond float64         `json:"tokens_per_second,omitempty"`
}

type chatStreamDelta struct {
	Role             string `json:"role,omitempty"`
	Content          string `json:"content,omitempty"`
	ReasoningContent string `json:"reasoning_content,omitempty"`
}
