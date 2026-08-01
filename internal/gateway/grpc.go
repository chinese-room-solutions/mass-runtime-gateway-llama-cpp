package gateway

import (
	"context"
	"errors"
	"fmt"
	"path/filepath"

	"github.com/KernelPryanic/ctxerr"
	llamacpp "github.com/chinese-room-solutions/mass-proto/gen/go/llama-cpp"
	workerpb "github.com/chinese-room-solutions/mass-proto/gen/go/worker"
	llamacppv1 "github.com/chinese-room-solutions/mass-runtime-gateway-llama-cpp/gen/go/llama_cpp/v1"
	"github.com/chinese-room-solutions/mass-runtime-gateway-llama-cpp/internal/model"
	"github.com/chinese-room-solutions/mass-runtime-gateway-llama-cpp/internal/payload"
	"github.com/chinese-room-solutions/mass-runtime-gateway-llama-cpp/internal/sched"
	"github.com/google/uuid"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// resolve validates a model path and builds its load artifacts + model id,
// returning gRPC status errors on bad input. Shared by the Submit handlers.
func (s *grpcServer) resolve(modelStr string, cfg *llamacppv1.ConfigOverride, kind llamacpp.LoadKind) (string, *llamacpp.LoadHints, []*workerpb.ModelFile, error) {
	storePath, err := s.h.resolveStorePath(modelStr)
	if err != nil {
		return "", nil, nil, status.Error(codes.InvalidArgument, err.Error())
	}
	hints, files, err := s.h.buildLoadArtifacts(modelStr, configFromOverride(cfg), kind)
	if err != nil {
		return "", nil, nil, status.Errorf(codes.InvalidArgument, "build load artifacts: %v", err)
	}
	return model.ID(storePath, hints), hints, files, nil
}

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

// --- Submit (enqueue, return job id) ---

// submitJob is the shared gRPC submit path: build → SubmitOnly → {job_id}. The
// build closure owns request validation + job/artifact assembly and returns a
// gRPC status error on bad input (pre-schedule failures stay honest codes).
func (s *grpcServer) submitJob(ctx context.Context, build func() (*llamacpp.Job, string, *llamacpp.LoadHints, []*workerpb.ModelFile, error)) (*llamacppv1.SubmitResponse, error) {
	job, modelID, hints, files, err := build()
	if err != nil {
		return nil, err
	}
	params, err := s.h.buildScheduleParams(ctx, modelID, job, hints, files)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "build schedule params: %v", err)
	}
	jobID, err := s.h.scheduler.SubmitOnly(ctx, params)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "submit: %v", err)
	}
	return &llamacppv1.SubmitResponse{JobId: jobID}, nil
}

func (s *grpcServer) SubmitChat(ctx context.Context, req *llamacppv1.ChatRequest) (*llamacppv1.SubmitResponse, error) {
	return s.submitJob(ctx, func() (*llamacpp.Job, string, *llamacpp.LoadHints, []*workerpb.ModelFile, error) {
		if req.GetModel() == "" {
			return nil, "", nil, nil, status.Error(codes.InvalidArgument, "model is required")
		}
		modelID, hints, files, err := s.resolve(req.GetModel(), req.GetModelConfig(), llamacpp.LoadKind_LOAD_KIND_CHAT)
		if err != nil {
			return nil, "", nil, nil, err
		}
		job := chatJobFromMessages(messagesFromProto(req.GetMessages()), samplingFromProto(req.GetSampling()), false)
		return job, modelID, hints, files, nil
	})
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
	jobID, chunks, err := s.h.dispatchWithID(stream.Context(), modelID, job, hints, files)
	if err != nil {
		return status.Errorf(codes.Internal, "dispatch: %v", err)
	}
	warnDecode := s.h.decodeWarner(jobID)

	id := "chatcmpl-" + uuid.NewString()
	for c := range chunks {
		switch c.Type {
		case sched.ChunkBody:
			dec, decErr := payload.DecodeJobChunk(c.Chunk)
			if decErr != nil {
				warnDecode(decErr)
				continue
			}
			delta := dec.GetChat()
			if delta == nil {
				continue
			}
			if err := stream.Send(&llamacppv1.ChatChunk{
				Id:             id,
				Model:          req.GetModel(),
				Delta:          &llamacppv1.Message{Role: roleToProto(delta.GetRole()), Content: delta.GetContent()},
				ReasoningDelta: delta.GetReasoningContent(),
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

func (s *grpcServer) SubmitBatchChat(ctx context.Context, req *llamacppv1.BatchChatRequest) (*llamacppv1.SubmitResponse, error) {
	return s.submitJob(ctx, func() (*llamacpp.Job, string, *llamacpp.LoadHints, []*workerpb.ModelFile, error) {
		if len(req.GetItems()) == 0 {
			return nil, "", nil, nil, status.Error(codes.InvalidArgument, "items must be non-empty")
		}
		modelID, hints, files, err := s.resolve(req.GetModel(), req.GetModelConfig(), llamacpp.LoadKind_LOAD_KIND_CHAT)
		if err != nil {
			return nil, "", nil, nil, err
		}
		items := make([]*llamacpp.BatchChatItem, len(req.GetItems()))
		for i, it := range req.GetItems() {
			items[i] = &llamacpp.BatchChatItem{
				Messages: chatMessagesToProto(messagesFromProto(it.GetMessages())),
				Sampling: convertSampling(samplingFromProto(it.GetSampling())),
			}
		}
		job := &llamacpp.Job{
			Kind: llamacpp.JobKind_JOB_KIND_BATCH_CHAT,
			Body: &llamacpp.Job_BatchChat{BatchChat: &llamacpp.BatchChatJob{Items: items}},
		}
		return job, modelID, hints, files, nil
	})
}

func (s *grpcServer) SubmitEmbed(ctx context.Context, req *llamacppv1.EmbedRequest) (*llamacppv1.SubmitResponse, error) {
	return s.submitJob(ctx, func() (*llamacpp.Job, string, *llamacpp.LoadHints, []*workerpb.ModelFile, error) {
		modelID, hints, files, err := s.resolve(req.GetModel(), req.GetModelConfig(), llamacpp.LoadKind_LOAD_KIND_EMBEDDING)
		if err != nil {
			return nil, "", nil, nil, err
		}
		job := &llamacpp.Job{
			Kind: llamacpp.JobKind_JOB_KIND_EMBED,
			Body: &llamacpp.Job_Embed{Embed: &llamacpp.EmbedJob{Input: req.GetInput()}},
		}
		return job, modelID, hints, files, nil
	})
}

func (s *grpcServer) SubmitBatchEmbed(ctx context.Context, req *llamacppv1.BatchEmbedRequest) (*llamacppv1.SubmitResponse, error) {
	return s.submitJob(ctx, func() (*llamacpp.Job, string, *llamacpp.LoadHints, []*workerpb.ModelFile, error) {
		modelID, hints, files, err := s.resolve(req.GetModel(), req.GetModelConfig(), llamacpp.LoadKind_LOAD_KIND_EMBEDDING)
		if err != nil {
			return nil, "", nil, nil, err
		}
		job := &llamacpp.Job{
			Kind: llamacpp.JobKind_JOB_KIND_BATCH_EMBED,
			Body: &llamacpp.Job_BatchEmbed{BatchEmbed: &llamacpp.BatchEmbedJob{Inputs: req.GetInputs()}},
		}
		return job, modelID, hints, files, nil
	})
}

func (s *grpcServer) SubmitTokenize(ctx context.Context, req *llamacppv1.TokenizeRequest) (*llamacppv1.SubmitResponse, error) {
	return s.submitJob(ctx, func() (*llamacpp.Job, string, *llamacpp.LoadHints, []*workerpb.ModelFile, error) {
		// Tokenize uses the chat tokenizer; same load kind as chat.
		modelID, hints, files, err := s.resolve(req.GetModel(), req.GetModelConfig(), llamacpp.LoadKind_LOAD_KIND_CHAT)
		if err != nil {
			return nil, "", nil, nil, err
		}
		job := &llamacpp.Job{
			Kind: llamacpp.JobKind_JOB_KIND_TOKENIZE,
			Body: &llamacpp.Job_Tokenize{Tokenize: &llamacpp.TokenizeJob{Text: req.GetText()}},
		}
		return job, modelID, hints, files, nil
	})
}

// --- Fetch / cancel ---

// GetResult reads a submitted job's result by id. wait=true drains the reattach
// stream to terminal (durable read — a client disconnect ends the drain but
// never cancels the job); wait=false returns the current status without
// blocking. The stored terminal chunk self-describes its type, so one method
// serves chat/batch-chat/embed/batch-embed/tokenize.
func (s *grpcServer) GetResult(ctx context.Context, req *llamacppv1.GetResultRequest) (*llamacppv1.JobResult, error) {
	id := req.GetJobId()
	if id == "" {
		return nil, status.Error(codes.InvalidArgument, "job_id is required")
	}
	if req.GetWait() {
		for range s.h.scheduler.Reattach(ctx, id) { //nolint:revive // drain to terminal; result read below
		}
		if ctx.Err() != nil {
			return nil, status.FromContextError(ctx.Err()).Err()
		}
	}
	res, err := s.h.scheduler.GetResult(ctx, id)
	if err != nil {
		if errors.Is(err, sched.ErrResultNotFound) {
			return nil, status.Errorf(codes.NotFound, "job %q not found or expired", id)
		}
		return nil, status.Errorf(codes.Internal, "get result: %v", err)
	}

	out := &llamacppv1.JobResult{Status: jobStatusToProto(res.Status)}
	switch res.Status {
	case sched.ResultError:
		out.Error = res.Err
	case sched.ResultDone:
		if err := applyJobResultProto(out, res.Body); err != nil {
			return nil, status.Errorf(codes.Internal, "decode result: %v", err)
		}
	}
	return out, nil
}

// CancelJob cancels a submitted job (pending or running) by id.
func (s *grpcServer) CancelJob(ctx context.Context, req *llamacppv1.CancelJobRequest) (*llamacppv1.CancelJobResponse, error) {
	id := req.GetJobId()
	if id == "" {
		return nil, status.Error(codes.InvalidArgument, "job_id is required")
	}
	if err := s.h.scheduler.CancelJob(ctx, id); err != nil {
		if errors.Is(err, sched.ErrResultNotFound) {
			return nil, status.Errorf(codes.NotFound, "job %q not cancellable (finished, unknown, or expired)", id)
		}
		return nil, status.Errorf(codes.Internal, "cancel job: %v", err)
	}
	return &llamacppv1.CancelJobResponse{}, nil
}

// applyJobResultProto fills the JobResult oneof from a stored terminal JobChunk,
// switching on its self-describing type — the proto mirror of decodeJobResult.
func applyJobResultProto(out *llamacppv1.JobResult, body []byte) error {
	if len(body) == 0 {
		return nil
	}
	dec, err := payload.DecodeJobChunk(body)
	if err != nil {
		return ctxerr.With(fmt.Errorf("decoding job result: %w", err), nil)
	}
	switch {
	case dec.GetChatFinal() != nil:
		resp := &llamacppv1.ChatResponse{Id: "chatcmpl-" + uuid.NewString()}
		applyChatFinalProto(resp, dec.GetChatFinal())
		out.Result = &llamacppv1.JobResult_Chat{Chat: resp}
	case dec.GetBatchChat() != nil:
		br := dec.GetBatchChat()
		resp := &llamacppv1.BatchChatResponse{Responses: make([]*llamacppv1.ChatResponse, len(br.GetItems()))}
		for i, cf := range br.GetItems() {
			resp.Responses[i] = &llamacppv1.ChatResponse{Id: cf.GetId()}
			applyChatFinalProto(resp.Responses[i], cf)
		}
		out.Result = &llamacppv1.JobResult_BatchChat{BatchChat: resp}
	case dec.GetEmbed() != nil:
		out.Result = &llamacppv1.JobResult_Embed{Embed: &llamacppv1.EmbedResponse{Embedding: dec.GetEmbed().GetEmbedding()}}
	case dec.GetBatchEmbed() != nil:
		br := dec.GetBatchEmbed()
		resp := &llamacppv1.BatchEmbedResponse{Embeddings: make([]*llamacppv1.EmbeddingItem, len(br.GetItems()))}
		for i, it := range br.GetItems() {
			resp.Embeddings[i] = &llamacppv1.EmbeddingItem{Index: it.GetIndex(), Embedding: it.GetEmbedding()}
		}
		out.Result = &llamacppv1.JobResult_BatchEmbed{BatchEmbed: resp}
	case dec.GetTokenize() != nil:
		out.Result = &llamacppv1.JobResult_Tokenize{Tokenize: &llamacppv1.TokenizeResponse{Tokens: dec.GetTokenize().GetTokens()}}
	}
	return nil
}

// jobStatusToProto maps the scheduler's result status to the proto enum.
func jobStatusToProto(s sched.ResultStatus) llamacppv1.JobStatus {
	switch s {
	case sched.ResultProcessing:
		return llamacppv1.JobStatus_JOB_STATUS_PROCESSING
	case sched.ResultDone:
		return llamacppv1.JobStatus_JOB_STATUS_DONE
	case sched.ResultError:
		return llamacppv1.JobStatus_JOB_STATUS_ERROR
	default:
		return llamacppv1.JobStatus_JOB_STATUS_PENDING
	}
}

// --- List ---

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
		ContextSize:     o.GetContextSize(),
		BatchSize:       optionalPtr(o.BatchSize),
		GPULayers:       optionalPtr(o.GpuLayers),
		FlashAttn:       optionalPtr(o.FlashAttn),
		Threads:         optionalPtr(o.Threads),
		MaxConcurrent:   optionalPtr(o.MaxConcurrent),
		VramHeadroomPct: optionalPtr(o.VramHeadroomPct),
		Thinking:        o.GetThinking(),
		MmprojFilename:  mmprojBaseFromProto(o.GetMmprojFilename()),
		ChatTemplate:    o.GetChatTemplate(),
		CacheType:       cacheTypeFromProto(o.GetCacheType()),
	}
}

// mmprojBaseFromProto strips any directory components from the proto
// field, but returns "" for the empty input — filepath.Base("") returns
// ".", which downstream code (handlers.go:buildLoadArtifacts) interprets
// as "mmproj filename set" and tries to load the model's directory as
// a CLIP model.
func mmprojBaseFromProto(s string) string {
	if s == "" {
		return ""
	}
	return filepath.Base(s)
}

// optionalPtr copies an optional proto field so downstream structs never
// alias proto message memory.
func optionalPtr[T any](p *T) *T {
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
		MaxTokens:        optionalPtr(s.MaxTokens),
		Temperature:      optionalPtr(s.Temperature),
		TopP:             optionalPtr(s.TopP),
		TopK:             optionalPtr(s.TopK),
		Seed:             optionalPtr(s.Seed),
		Stop:             s.GetStop(),
		MinP:             optionalPtr(s.MinP),
		RepeatPenalty:    optionalPtr(s.RepeatPenalty),
		FrequencyPenalty: optionalPtr(s.FrequencyPenalty),
		PresencePenalty:  optionalPtr(s.PresencePenalty),
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
