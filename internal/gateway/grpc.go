package gateway

import (
	"context"
	"fmt"
	"path/filepath"

	"github.com/KernelPryanic/ctxerr"
	llamacpp "github.com/chinese-room-solutions/mass-proto/gen/go/llama-cpp"
	llamacppv1 "github.com/chinese-room-solutions/mass-runtime-llama-cpp/gen/go/llama_cpp/v1"
	"github.com/chinese-room-solutions/mass-runtime-llama-cpp/internal/model"
	"github.com/chinese-room-solutions/mass-runtime-llama-cpp/internal/payload"
	"github.com/chinese-room-solutions/mass-runtime-llama-cpp/internal/sched"
	"github.com/google/uuid"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// grpcServer implements llama_cpp.v1.LlamaCppService. It shares the gateway's
// dispatch helpers (h.scheduler, h.buildLoadArtifacts, ...) so the typed gRPC
// API and the typed HTTP/JSON API serve from one source of truth — the gRPC
// methods just shoulder request/response conversion.
type grpcServer struct {
	llamacppv1.UnimplementedLlamaCppServiceServer
	h *handlers
}

func newGRPCServer(h *handlers) *grpcServer {
	return &grpcServer{h: h}
}

// --- Chat ---

func (s *grpcServer) Chat(ctx context.Context, req *llamacppv1.ChatRequest) (*llamacppv1.ChatResponse, error) {
	if req.GetModel() == "" {
		return nil, status.Error(codes.InvalidArgument, "model is required")
	}
	storePath := model.ResolveModelPath(req.GetModel())
	if storePath == "" {
		return nil, status.Errorf(codes.InvalidArgument, "model: invalid path %q", req.GetModel())
	}
	hints, files, err := s.h.buildLoadArtifacts(req.GetModel(), configFromOverride(req.GetModelConfig()), llamacpp.LoadKind_LOAD_KIND_CHAT)
	if err != nil {
		return nil, status.Errorf(codes.InvalidArgument, "build load artifacts: %v", err)
	}
	modelID := model.ID(storePath, hints)

	job := chatJobFromMessages(messagesFromProto(req.GetMessages()), samplingFromProto(req.GetSampling()), false)
	chunks, err := s.h.dispatch(ctx, modelID, job, hints, files)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "dispatch: %v", err)
	}

	resp := &llamacppv1.ChatResponse{
		Id:    "chatcmpl-" + uuid.NewString(),
		Model: req.GetModel(),
	}
	for c := range chunks {
		switch c.Type {
		case sched.ChunkCompleted:
			if len(c.Final) == 0 {
				continue
			}
			dec, decErr := payload.DecodeJobChunk(c.Final)
			if decErr != nil {
				return nil, status.Errorf(codes.Internal, "decode final chunk: %v", decErr)
			}
			cf := dec.GetChatFinal()
			if cf == nil {
				continue
			}
			applyChatFinalProto(resp, cf)
		case sched.ChunkError:
			return nil, status.Errorf(codes.Internal, "worker: %s", c.ErrText)
		}
	}
	return resp, nil
}

func (s *grpcServer) ChatStream(req *llamacppv1.ChatRequest, stream llamacppv1.LlamaCppService_ChatStreamServer) error {
	if req.GetModel() == "" {
		return status.Error(codes.InvalidArgument, "model is required")
	}
	storePath := model.ResolveModelPath(req.GetModel())
	if storePath == "" {
		return status.Errorf(codes.InvalidArgument, "model: invalid path %q", req.GetModel())
	}
	hints, files, err := s.h.buildLoadArtifacts(req.GetModel(), configFromOverride(req.GetModelConfig()), llamacpp.LoadKind_LOAD_KIND_CHAT)
	if err != nil {
		return status.Errorf(codes.InvalidArgument, "build load artifacts: %v", err)
	}
	modelID := model.ID(storePath, hints)

	job := chatJobFromMessages(messagesFromProto(req.GetMessages()), samplingFromProto(req.GetSampling()), true)
	chunks, err := s.h.dispatch(stream.Context(), modelID, job, hints, files)
	if err != nil {
		return status.Errorf(codes.Internal, "dispatch: %v", err)
	}

	id := "chatcmpl-" + uuid.NewString()
	for c := range chunks {
		switch c.Type {
		case sched.ChunkBody:
			dec, decErr := payload.DecodeJobChunk(c.Chunk)
			if decErr != nil {
				continue
			}
			delta := dec.GetChat()
			if delta == nil {
				continue
			}
			if err := stream.Send(&llamacppv1.ChatChunk{
				Id:              id,
				Model:           req.GetModel(),
				Delta:           &llamacppv1.Message{Role: roleToProto(delta.GetRole()), Content: delta.GetContent()},
				ReasoningDelta:  delta.GetReasoningContent(),
			}); err != nil {
				return ctxerr.With(fmt.Errorf("sending chat chunk: %w", err), nil)
			}
		case sched.ChunkCompleted:
			if len(c.Final) == 0 {
				continue
			}
			dec, decErr := payload.DecodeJobChunk(c.Final)
			if decErr != nil {
				return status.Errorf(codes.Internal, "decode final chunk: %v", decErr)
			}
			cf := dec.GetChatFinal()
			if cf == nil {
				continue
			}
			// Workers that don't stream tokens (today: every backend) deliver
			// the full assistant text on the terminal frame's `message`.
			// Re-emit it as one body chunk so streaming clients still see the
			// content even when the upstream response was synchronous.
			if msg := cf.GetMessage(); msg != nil && msg.GetContent() != "" {
				if err := stream.Send(&llamacppv1.ChatChunk{
					Id:    id,
					Model: req.GetModel(),
					Delta: &llamacppv1.Message{
						Role:    roleToProto(msg.GetRole()),
						Content: msg.GetContent(),
					},
					ReasoningDelta: cf.GetReasoningContent(),
				}); err != nil {
					return ctxerr.With(fmt.Errorf("sending body chunk from final: %w", err), nil)
				}
			}
			final := &llamacppv1.ChatChunk{
				Id:              id,
				Model:           req.GetModel(),
				FinishReason:    finishReasonToProto(cf.GetFinishReason()),
				TokensPerSecond: cf.GetTokensPerSecond(),
			}
			if u := cf.GetUsage(); u != nil {
				final.Usage = &llamacppv1.Usage{
					PromptTokens:     u.GetPromptTokens(),
					CompletionTokens: u.GetCompletionTokens(),
					TotalTokens:      u.GetTotalTokens(),
				}
			}
			if err := stream.Send(final); err != nil {
				return ctxerr.With(fmt.Errorf("sending final chunk: %w", err), nil)
			}
		case sched.ChunkError:
			return status.Errorf(codes.Internal, "worker: %s", c.ErrText)
		}
	}
	return nil
}

// --- Batch chat ---

func (s *grpcServer) BatchChat(ctx context.Context, req *llamacppv1.BatchChatRequest) (*llamacppv1.BatchChatResponse, error) {
	if len(req.GetItems()) == 0 {
		return nil, status.Error(codes.InvalidArgument, "items must be non-empty")
	}
	storePath := model.ResolveModelPath(req.GetModel())
	if storePath == "" {
		return nil, status.Errorf(codes.InvalidArgument, "model: invalid path %q", req.GetModel())
	}
	hints, files, err := s.h.buildLoadArtifacts(req.GetModel(), configFromOverride(req.GetModelConfig()), llamacpp.LoadKind_LOAD_KIND_CHAT)
	if err != nil {
		return nil, status.Errorf(codes.InvalidArgument, "build load artifacts: %v", err)
	}
	modelID := model.ID(storePath, hints)

	out := make([]*llamacppv1.ChatResponse, len(req.GetItems()))
	for i, item := range req.GetItems() {
		job := chatJobFromMessages(messagesFromProto(item.GetMessages()), samplingFromProto(item.GetSampling()), false)
		chunks, err := s.h.dispatch(ctx, modelID, job, hints, files)
		if err != nil {
			return nil, status.Errorf(codes.Internal, "item %d dispatch: %v", i, err)
		}
		resp := &llamacppv1.ChatResponse{
			Id:    "chatcmpl-" + uuid.NewString(),
			Model: req.GetModel(),
		}
		for c := range chunks {
			switch c.Type {
			case sched.ChunkCompleted:
				if len(c.Final) == 0 {
					continue
				}
				if dec, decErr := payload.DecodeJobChunk(c.Final); decErr == nil {
					if cf := dec.GetChatFinal(); cf != nil {
						applyChatFinalProto(resp, cf)
					}
				}
			case sched.ChunkError:
				return nil, status.Errorf(codes.Internal, "item %d worker: %s", i, c.ErrText)
			}
		}
		out[i] = resp
	}
	return &llamacppv1.BatchChatResponse{Responses: out}, nil
}

// --- Embed ---

func (s *grpcServer) Embed(ctx context.Context, req *llamacppv1.EmbedRequest) (*llamacppv1.EmbedResponse, error) {
	storePath := model.ResolveModelPath(req.GetModel())
	if storePath == "" {
		return nil, status.Errorf(codes.InvalidArgument, "model: invalid path %q", req.GetModel())
	}
	hints, files, err := s.h.buildLoadArtifacts(req.GetModel(), configFromOverride(req.GetModelConfig()), llamacpp.LoadKind_LOAD_KIND_EMBEDDING)
	if err != nil {
		return nil, status.Errorf(codes.InvalidArgument, "build load artifacts: %v", err)
	}
	modelID := model.ID(storePath, hints)

	job := &llamacpp.Job{
		Kind: llamacpp.JobKind_JOB_KIND_EMBED,
		Body: &llamacpp.Job_Embed{Embed: &llamacpp.EmbedJob{Input: req.GetInput()}},
	}
	chunks, err := s.h.dispatch(ctx, modelID, job, hints, files)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "dispatch: %v", err)
	}
	resp := &llamacppv1.EmbedResponse{Id: uuid.NewString(), Model: req.GetModel()}
	for c := range chunks {
		switch c.Type {
		case sched.ChunkCompleted:
			if len(c.Final) == 0 {
				continue
			}
			if dec, decErr := payload.DecodeJobChunk(c.Final); decErr == nil {
				if er := dec.GetEmbed(); er != nil {
					resp.Embedding = er.GetEmbedding()
				}
			}
		case sched.ChunkError:
			return nil, status.Errorf(codes.Internal, "worker: %s", c.ErrText)
		}
	}
	return resp, nil
}

func (s *grpcServer) BatchEmbed(ctx context.Context, req *llamacppv1.BatchEmbedRequest) (*llamacppv1.BatchEmbedResponse, error) {
	storePath := model.ResolveModelPath(req.GetModel())
	if storePath == "" {
		return nil, status.Errorf(codes.InvalidArgument, "model: invalid path %q", req.GetModel())
	}
	hints, files, err := s.h.buildLoadArtifacts(req.GetModel(), configFromOverride(req.GetModelConfig()), llamacpp.LoadKind_LOAD_KIND_EMBEDDING)
	if err != nil {
		return nil, status.Errorf(codes.InvalidArgument, "build load artifacts: %v", err)
	}
	modelID := model.ID(storePath, hints)

	job := &llamacpp.Job{
		Kind: llamacpp.JobKind_JOB_KIND_BATCH_EMBED,
		Body: &llamacpp.Job_BatchEmbed{BatchEmbed: &llamacpp.BatchEmbedJob{Inputs: req.GetInputs()}},
	}
	chunks, err := s.h.dispatch(ctx, modelID, job, hints, files)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "dispatch: %v", err)
	}
	resp := &llamacppv1.BatchEmbedResponse{Id: uuid.NewString(), Model: req.GetModel()}
	for c := range chunks {
		switch c.Type {
		case sched.ChunkCompleted:
			if len(c.Final) == 0 {
				continue
			}
			if dec, decErr := payload.DecodeJobChunk(c.Final); decErr == nil {
				if br := dec.GetBatchEmbed(); br != nil {
					resp.Embeddings = make([]*llamacppv1.EmbeddingItem, len(br.GetItems()))
					for i, it := range br.GetItems() {
						resp.Embeddings[i] = &llamacppv1.EmbeddingItem{Index: it.GetIndex(), Embedding: it.GetEmbedding()}
					}
				}
			}
		case sched.ChunkError:
			return nil, status.Errorf(codes.Internal, "worker: %s", c.ErrText)
		}
	}
	return resp, nil
}

// --- Tokenize ---

func (s *grpcServer) Tokenize(ctx context.Context, req *llamacppv1.TokenizeRequest) (*llamacppv1.TokenizeResponse, error) {
	storePath := model.ResolveModelPath(req.GetModel())
	if storePath == "" {
		return nil, status.Errorf(codes.InvalidArgument, "model: invalid path %q", req.GetModel())
	}
	// Tokenize uses the chat tokenizer; same load kind as chat.
	hints, files, err := s.h.buildLoadArtifacts(req.GetModel(), configFromOverride(req.GetModelConfig()), llamacpp.LoadKind_LOAD_KIND_CHAT)
	if err != nil {
		return nil, status.Errorf(codes.InvalidArgument, "build load artifacts: %v", err)
	}
	modelID := model.ID(storePath, hints)
	job := &llamacpp.Job{
		Kind: llamacpp.JobKind_JOB_KIND_TOKENIZE,
		Body: &llamacpp.Job_Tokenize{Tokenize: &llamacpp.TokenizeJob{Text: req.GetText()}},
	}
	chunks, err := s.h.dispatch(ctx, modelID, job, hints, files)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "dispatch: %v", err)
	}
	resp := &llamacppv1.TokenizeResponse{}
	for c := range chunks {
		switch c.Type {
		case sched.ChunkCompleted:
			if len(c.Final) == 0 {
				continue
			}
			if dec, decErr := payload.DecodeJobChunk(c.Final); decErr == nil {
				if tr := dec.GetTokenize(); tr != nil {
					resp.Tokens = tr.GetTokens()
				}
			}
		case sched.ChunkError:
			return nil, status.Errorf(codes.Internal, "worker: %s", c.ErrText)
		}
	}
	return resp, nil
}

// --- Load / list ---

func (s *grpcServer) LoadModel(ctx context.Context, req *llamacppv1.LoadModelRequest) (*llamacppv1.LoadModelResponse, error) {
	storePath := model.ResolveModelPath(req.GetModel())
	if storePath == "" {
		return nil, status.Errorf(codes.InvalidArgument, "model: invalid path %q", req.GetModel())
	}
	kind := llamacpp.LoadKind_LOAD_KIND_CHAT
	if req.GetKind() == llamacppv1.LoadKind_LOAD_KIND_EMBEDDING {
		kind = llamacpp.LoadKind_LOAD_KIND_EMBEDDING
	}
	hints, files, err := s.h.buildLoadArtifacts(req.GetModel(), configFromOverride(req.GetModelConfig()), kind)
	if err != nil {
		return nil, status.Errorf(codes.InvalidArgument, "build load artifacts: %v", err)
	}
	modelID := model.ID(storePath, hints)
	hintsBytes, err := payload.EncodeLoadHints(hints)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "encode hints: %v", err)
	}
	instances, err := s.h.scheduler.EnsureModelLoaded(ctx, sched.EnsureModelLoadedParams{
		ModelID:   modelID,
		Files:     files,
		LoadHints: hintsBytes,
		Source:    sourceFromContext(ctx),
	})
	if err != nil {
		return nil, status.Errorf(codes.Internal, "ensure loaded: %v", err)
	}
	if len(instances) == 0 {
		return nil, status.Error(codes.Internal, "scheduler returned no instances")
	}
	return &llamacppv1.LoadModelResponse{
		ModelId:  modelID,
		WorkerId: instances[0].WorkerID,
		PoolSize: instances[0].PoolSize,
	}, nil
}

// ListModels is the typed app-facing model catalog. It walks the gateway's
// models_dir using the same shared [parseModelInfo] cache as the
// MASS-facing [Gateway.ListModels], then projects the internal parsed view
// into the slimmer typed shape the LlamaCppService surface exposes.
// Companion files (mmproj projectors) are filtered out — apps want
// user-selectable chat/embedding models only.
func (s *grpcServer) ListModels(ctx context.Context, _ *llamacppv1.ListModelsRequest) (*llamacppv1.ListModelsResponse, error) {
	_ = ctx
	infos, err := s.h.cache.walkAndParseModels(formatRoot(s.h.modelsDir))
	if err != nil {
		return nil, status.Errorf(codes.Internal, "list models: %v", err)
	}
	out := make([]*llamacppv1.ModelEntry, 0, len(infos))
	for _, info := range infos {
		if info.Companion != "" {
			continue
		}
		caps := info.Capabilities
		out = append(out, &llamacppv1.ModelEntry{
			Id:        info.ID,
			SizeBytes: info.SizeBytes,
			Capabilities: &llamacppv1.ModelCapabilities{
				Vision:   caps.GetVision(),
				Audio:    caps.GetAudio(),
				Thinking: caps.GetThinking(),
			},
			Quantization: info.VariantLabel,
			ModelType:    info.ModelType,
		})
	}
	return &llamacppv1.ListModelsResponse{Models: out}, nil
}

// --- Conversions: typed gRPC <-> internal types ---

func configFromOverride(o *llamacppv1.ConfigOverride) *loadConfig {
	if o == nil {
		return nil
	}
	return &loadConfig{
		ContextSize:    o.GetContextSize(),
		BatchSize:      optionalInt32(o.BatchSize),
		GPULayers:      optionalInt32(o.GpuLayers),
		FlashAttn:      optionalBool(o.FlashAttn),
		Threads:        optionalInt32(o.Threads),
		MaxConcurrent:  optionalInt32(o.MaxConcurrent),
		Thinking:       o.GetThinking(),
		MainGPU:        o.GetMainGpu(),
		TensorSplit:    o.GetTensorSplit(),
		MmprojFilename: filepath.Base(o.GetMmprojFilename()),
		ChatTemplate:   o.GetChatTemplate(),
		CacheType:      cacheTypeFromProto(o.GetCacheType()),
	}
}

func optionalInt32(p *int32) *int32 {
	if p == nil {
		return nil
	}
	v := *p
	return &v
}

func optionalBool(p *bool) *bool {
	if p == nil {
		return nil
	}
	v := *p
	return &v
}

func cacheTypeFromProto(c llamacppv1.CacheType) string {
	switch c {
	case llamacppv1.CacheType_CACHE_TYPE_F16:
		return "f16"
	case llamacppv1.CacheType_CACHE_TYPE_Q8_0:
		return "q8_0"
	case llamacppv1.CacheType_CACHE_TYPE_Q4_0:
		return "q4_0"
	default:
		return ""
	}
}

func messagesFromProto(in []*llamacppv1.Message) []chatMessage {
	out := make([]chatMessage, len(in))
	for i, m := range in {
		out[i] = chatMessage{
			Role:    roleFromProto(m.GetRole()),
			Content: m.GetContent(),
			Parts:   partsFromProto(m.GetParts()),
		}
	}
	return out
}

func partsFromProto(in []*llamacppv1.ContentPart) []contentPart {
	if len(in) == 0 {
		return nil
	}
	out := make([]contentPart, len(in))
	for i, p := range in {
		out[i] = contentPart{
			Type:     p.GetType(),
			Text:     p.GetText(),
			Data:     p.GetData(),
			MIMEType: p.GetMimeType(),
		}
	}
	return out
}

func samplingFromProto(s *llamacppv1.Sampling) *samplingParams {
	if s == nil {
		return nil
	}
	return &samplingParams{
		MaxTokens:        optionalInt32(s.MaxTokens),
		Temperature:      s.GetTemperature(),
		TopP:             s.GetTopP(),
		TopK:             s.GetTopK(),
		Seed:             optionalInt32(s.Seed),
		Stop:             s.GetStop(),
		MinP:             s.GetMinP(),
		RepeatPenalty:    s.GetRepeatPenalty(),
		FrequencyPenalty: s.GetFrequencyPenalty(),
		PresencePenalty:  s.GetPresencePenalty(),
		EnableThinking:   s.GetEnableThinking(),
	}
}

func roleFromProto(r llamacppv1.Role) string {
	switch r {
	case llamacppv1.Role_ROLE_SYSTEM:
		return "system"
	case llamacppv1.Role_ROLE_USER:
		return "user"
	case llamacppv1.Role_ROLE_ASSISTANT:
		return "assistant"
	case llamacppv1.Role_ROLE_TOOL:
		return "tool"
	default:
		return ""
	}
}

func roleToProto(r llamacpp.Role) llamacppv1.Role {
	switch r {
	case llamacpp.Role_ROLE_SYSTEM:
		return llamacppv1.Role_ROLE_SYSTEM
	case llamacpp.Role_ROLE_USER:
		return llamacppv1.Role_ROLE_USER
	case llamacpp.Role_ROLE_ASSISTANT:
		return llamacppv1.Role_ROLE_ASSISTANT
	case llamacpp.Role_ROLE_TOOL:
		return llamacppv1.Role_ROLE_TOOL
	default:
		return llamacppv1.Role_ROLE_UNSPECIFIED
	}
}

func finishReasonToProto(f llamacpp.FinishReason) llamacppv1.FinishReason {
	switch f {
	case llamacpp.FinishReason_FINISH_REASON_STOP:
		return llamacppv1.FinishReason_FINISH_REASON_STOP
	case llamacpp.FinishReason_FINISH_REASON_LENGTH:
		return llamacppv1.FinishReason_FINISH_REASON_LENGTH
	case llamacpp.FinishReason_FINISH_REASON_CONTENT_FILTER:
		return llamacppv1.FinishReason_FINISH_REASON_CONTENT_FILTER
	case llamacpp.FinishReason_FINISH_REASON_TOOL_CALLS:
		return llamacppv1.FinishReason_FINISH_REASON_TOOL_CALLS
	default:
		return llamacppv1.FinishReason_FINISH_REASON_UNSPECIFIED
	}
}

// applyChatFinalProto fills in a ChatResponse from a worker's ChatFinal frame.
// Mirrors applyChatFinal (HTTP variant) but writes typed proto fields instead
// of JSON-tagged Go structs.
func applyChatFinalProto(resp *llamacppv1.ChatResponse, cf *llamacpp.ChatFinal) {
	if m := cf.GetMessage(); m != nil {
		resp.Message = &llamacppv1.Message{
			Role:    roleToProto(m.GetRole()),
			Content: m.GetContent(),
		}
	}
	resp.FinishReason = finishReasonToProto(cf.GetFinishReason())
	resp.ReasoningContent = cf.GetReasoningContent()
	if u := cf.GetUsage(); u != nil {
		resp.Usage = &llamacppv1.Usage{
			PromptTokens:     u.GetPromptTokens(),
			CompletionTokens: u.GetCompletionTokens(),
			TotalTokens:      u.GetTotalTokens(),
		}
	}
	resp.TokensPerSecond = cf.GetTokensPerSecond()
}
