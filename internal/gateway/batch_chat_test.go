package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	gatewaypb "github.com/chinese-room-solutions/mass-proto/gen/go/gateway"
	llamacpp "github.com/chinese-room-solutions/mass-proto/gen/go/llama-cpp"
	"github.com/chinese-room-solutions/mass-runtime-gateway-llama-cpp/internal/payload"
	"github.com/chinese-room-solutions/mass-runtime-gateway-llama-cpp/internal/sched"
	"github.com/rs/zerolog"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/test/bufconn"
)

// modelsDirWithPrimary writes a stub primary gguf and returns a handlers
// pointed at it, mirroring the load_artifacts test setup so
// model.ResolveModelPath resolves "qwen3/primary.gguf".
func modelsDirWithPrimary(t *testing.T) (*handlers, string) {
	t.Helper()
	modelsDir := t.TempDir()
	root := filepath.Join(modelsDir, formatDir, "qwen3")
	require.NoError(t, os.MkdirAll(root, 0o755))
	require.NoError(t, os.WriteFile(filepath.Join(root, "primary.gguf"), make([]byte, 4096), 0o644))
	return &handlers{modelsDir: modelsDir, logger: zerolog.Nop()}, "qwen3/primary.gguf"
}

func TestBuildBatchChatJob(t *testing.T) {
	h, modelStr := modelsDirWithPrimary(t)

	t.Run("empty items -> error", func(t *testing.T) {
		_, _, _, _, err := h.buildBatchChatJob(&batchChatRequest{Model: modelStr})
		require.Error(t, err)
	})

	t.Run("invalid model -> error", func(t *testing.T) {
		_, _, _, _, err := h.buildBatchChatJob(&batchChatRequest{
			Model: "../escape",
			Items: []batchChatItem{{Messages: []chatMessage{{Role: "user", Content: "hi"}}}},
		})
		require.Error(t, err)
	})

	t.Run("valid -> single BATCH_CHAT job with mapped items", func(t *testing.T) {
		maxTok := int32(64)
		temp := float32(0.7)
		req := &batchChatRequest{
			Model: modelStr,
			Items: []batchChatItem{
				{Messages: []chatMessage{{Role: "system", Content: "sys"}, {Role: "user", Content: "one"}}},
				{
					Messages: []chatMessage{{Role: "user", Content: "two"}},
					Sampling: &samplingParams{MaxTokens: &maxTok, Temperature: &temp},
				},
			},
		}
		job, modelID, hints, files, err := h.buildBatchChatJob(req)
		require.NoError(t, err)
		require.NotEmpty(t, modelID)
		require.NotNil(t, hints)
		require.NotEmpty(t, files)

		require.Equal(t, llamacpp.JobKind_JOB_KIND_BATCH_CHAT, job.GetKind())
		bc := job.GetBatchChat()
		require.NotNil(t, bc, "body must be the batch_chat oneof variant")
		require.Len(t, bc.GetItems(), 2)

		// Item 0: two messages, no sampling.
		it0 := bc.GetItems()[0]
		require.Len(t, it0.GetMessages(), 2)
		require.Equal(t, llamacpp.Role_ROLE_SYSTEM, it0.GetMessages()[0].GetRole())
		require.Equal(t, "one", it0.GetMessages()[1].GetContent())
		require.Nil(t, it0.GetSampling())

		// Item 1: sampling carried through.
		it1 := bc.GetItems()[1]
		require.Len(t, it1.GetMessages(), 1)
		require.NotNil(t, it1.GetSampling())
		require.Equal(t, int32(64), it1.GetSampling().GetMaxTokens())
		require.InDelta(t, 0.7, it1.GetSampling().GetTemperature(), 1e-6)
	})
}

func TestBatchChatFinalsToResponses(t *testing.T) {
	br := &llamacpp.BatchChatResult{
		Id: "bchat-1",
		Items: []*llamacpp.ChatFinal{
			{
				Id:           "i0",
				Message:      &llamacpp.ChatMessage{Role: llamacpp.Role_ROLE_ASSISTANT, Content: "first"},
				FinishReason: llamacpp.FinishReason_FINISH_REASON_STOP,
				Usage:        &llamacpp.Usage{PromptTokens: 3, CompletionTokens: 5, TotalTokens: 8},
			},
			{
				Id:           "i1",
				Message:      &llamacpp.ChatMessage{Role: llamacpp.Role_ROLE_ASSISTANT, Content: "second"},
				FinishReason: llamacpp.FinishReason_FINISH_REASON_LENGTH,
			},
		},
	}
	out := batchChatFinalsToResponses(br)
	require.Len(t, out, 2)

	require.Equal(t, "i0", out[0].ID, "per-item id propagates from ChatFinal.id")
	require.Equal(t, "first", out[0].Message.Content)
	require.Equal(t, "stop", out[0].FinishReason)
	require.NotNil(t, out[0].Usage)
	require.Equal(t, int32(8), out[0].Usage.TotalTokens)

	require.Equal(t, "i1", out[1].ID)
	require.Equal(t, "second", out[1].Message.Content)
	require.Equal(t, "length", out[1].FinishReason)
}

func TestBatchChatFinalsToResponses_Empty(t *testing.T) {
	require.Empty(t, batchChatFinalsToResponses(&llamacpp.BatchChatResult{}))
}

// batchChatScheduler captures the Submit payload and replays a BatchChatResult
// terminal frame, so the sync handler's "one dispatch + decode" path runs
// end-to-end over bufconn.
type batchChatScheduler struct {
	gatewaypb.UnimplementedMassSchedulerServer
	submits    int
	streams    int
	gotKind    llamacpp.JobKind
	gotItems   int
	streamedID string
	resultIDs  []string
}

func (c *batchChatScheduler) Submit(_ context.Context, req *gatewaypb.SubmitRequest) (*gatewaypb.SubmitResponse, error) {
	c.submits++
	if job, err := payload.DecodeJob(req.GetPayload()); err == nil {
		c.gotKind = job.GetKind()
		c.gotItems = len(job.GetBatchChat().GetItems())
	}
	return &gatewaypb.SubmitResponse{JobId: "job-1"}, nil
}

func (c *batchChatScheduler) batchFinalBytes() ([]byte, error) {
	items := make([]*llamacpp.ChatFinal, len(c.resultIDs))
	for i, id := range c.resultIDs {
		items[i] = &llamacpp.ChatFinal{
			Id:           id,
			Message:      &llamacpp.ChatMessage{Role: llamacpp.Role_ROLE_ASSISTANT, Content: "resp-" + id},
			FinishReason: llamacpp.FinishReason_FINISH_REASON_STOP,
		}
	}
	return payload.EncodeJobChunk(&llamacpp.JobChunk{
		Body: &llamacpp.JobChunk_BatchChat{BatchChat: &llamacpp.BatchChatResult{Id: "bchat", Items: items}},
	})
}

func (c *batchChatScheduler) StreamChunks(req *gatewaypb.StreamChunksRequest, s gatewaypb.MassScheduler_StreamChunksServer) error {
	c.streams++
	c.streamedID = req.GetJobId()
	final, err := c.batchFinalBytes()
	if err != nil {
		return err
	}
	return s.Send(&gatewaypb.ScheduleChunk{
		JobId: "job-1",
		Frame: &gatewaypb.ScheduleChunk_Completed{
			Completed: &gatewaypb.ScheduleCompleted{FinalResponse: final, WorkerId: "w-1"},
		},
	})
}

func (c *batchChatScheduler) GetResult(_ context.Context, _ *gatewaypb.GetResultRequest) (*gatewaypb.GetResultResponse, error) {
	final, err := c.batchFinalBytes()
	if err != nil {
		return nil, err
	}
	return &gatewaypb.GetResultResponse{Status: gatewaypb.ResultStatus_RESULT_STATUS_DONE, Body: final}, nil
}

func newBatchChatTestClient(t *testing.T, server gatewaypb.MassSchedulerServer) (*sched.Client, func()) {
	t.Helper()
	lis := bufconn.Listen(1 << 16)
	grpcServer := grpc.NewServer()
	gatewaypb.RegisterMassSchedulerServer(grpcServer, server)
	var once sync.Once
	go func() {
		if err := grpcServer.Serve(lis); err != nil && !errors.Is(err, grpc.ErrServerStopped) {
			once.Do(func() { t.Logf("grpc.Serve: %v", err) })
		}
	}()
	conn, err := grpc.NewClient(
		"passthrough://bufnet",
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithContextDialer(func(_ context.Context, _ string) (net.Conn, error) { return lis.Dial() }),
	)
	require.NoError(t, err)
	cleanup := func() { _ = conn.Close(); grpcServer.Stop(); _ = lis.Close() }
	return sched.NewClient(conn, zerolog.Nop()), cleanup
}

// BatchChat submits once (all items in one job), returns the job id, and the
// result is read via GET /.v1/Jobs/{id}?wait=1 — one dispatch, decode the
// index-aligned responses.
func TestHandleBatchChat_SubmitThenWaitForResult(t *testing.T) {
	h, modelStr := modelsDirWithPrimary(t)

	fake := &batchChatScheduler{resultIDs: []string{"a", "b"}}
	client, cleanup := newBatchChatTestClient(t, fake)
	defer cleanup()
	h.scheduler = client

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	body, err := json.Marshal(batchChatRequest{
		Model: modelStr,
		Items: []batchChatItem{
			{Messages: []chatMessage{{Role: "user", Content: "one"}}},
			{Messages: []chatMessage{{Role: "user", Content: "two"}}},
		},
	})
	require.NoError(t, err)

	// Submit: one Submit (all items in one job), id returned, no result yet.
	submitRec := httptest.NewRecorder()
	submitReq := httptest.NewRequest(http.MethodPost, "/.v1/BatchChat", io.NopCloser(bytes.NewReader(body))).WithContext(ctx)
	h.handleBatchChat(submitRec, submitReq)
	require.Equal(t, http.StatusOK, submitRec.Code, submitRec.Body.String())
	require.Equal(t, 1, fake.submits, "batch chat must dispatch exactly once")
	require.Equal(t, llamacpp.JobKind_JOB_KIND_BATCH_CHAT, fake.gotKind)
	require.Equal(t, 2, fake.gotItems, "all items ride in one job")
	jobID := submitRec.Header().Get(jobIDHeader)
	require.Equal(t, "job-1", jobID)

	var sub submitResponse
	require.NoError(t, json.Unmarshal(submitRec.Body.Bytes(), &sub))
	require.Equal(t, "job-1", sub.JobID)

	// Fetch with ?wait=1: drains to terminal, decodes the batch responses.
	fetchRec := httptest.NewRecorder()
	fetchReq := httptest.NewRequest(http.MethodGet, "/.v1/Jobs/job-1?wait=1", nil).WithContext(ctx)
	fetchReq.SetPathValue("id", "job-1")
	h.handleJobResult(fetchRec, fetchReq)
	require.Equal(t, http.StatusOK, fetchRec.Code, fetchRec.Body.String())
	require.Equal(t, 1, fake.submits, "fetch must not Submit again")
	require.Equal(t, jobID, fake.streamedID, "fetch re-attaches the same job")

	var out jobResultResponse
	require.NoError(t, json.Unmarshal(fetchRec.Body.Bytes(), &out))
	require.Equal(t, "done", out.Status)
	var resp batchChatResponse
	require.NoError(t, json.Unmarshal(out.Result, &resp))
	require.Len(t, resp.Responses, 2)
	require.Equal(t, "resp-a", resp.Responses[0].Message.Content)
	require.Equal(t, "resp-b", resp.Responses[1].Message.Content)
}
