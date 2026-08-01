package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	gatewaypb "github.com/chinese-room-solutions/mass-proto/gen/go/gateway"
	llamacpp "github.com/chinese-room-solutions/mass-proto/gen/go/llama-cpp"
	"github.com/chinese-room-solutions/mass-runtime-gateway-llama-cpp/internal/payload"
	"github.com/stretchr/testify/require"
)

// chatScheduler records Submit/StreamChunks/GetResult and replays a ChatFinal
// terminal frame, so the submit + GET Jobs/{id}?wait=1 paths run end-to-end
// over bufconn.
type chatScheduler struct {
	gatewaypb.UnimplementedMassSchedulerServer
	submits     int
	streams     int
	streamedID  string
	content     string
	lastPayload []byte
}

func (c *chatScheduler) chatFinalBytes() ([]byte, error) {
	return payload.EncodeJobChunk(&llamacpp.JobChunk{
		Body: &llamacpp.JobChunk_ChatFinal{ChatFinal: &llamacpp.ChatFinal{
			Message:      &llamacpp.ChatMessage{Role: llamacpp.Role_ROLE_ASSISTANT, Content: c.content},
			FinishReason: llamacpp.FinishReason_FINISH_REASON_STOP,
		}},
	})
}

func (c *chatScheduler) Submit(_ context.Context, req *gatewaypb.SubmitRequest) (*gatewaypb.SubmitResponse, error) {
	c.submits++
	c.lastPayload = req.GetPayload()
	return &gatewaypb.SubmitResponse{JobId: "chat-job"}, nil
}

func (c *chatScheduler) StreamChunks(req *gatewaypb.StreamChunksRequest, s gatewaypb.MassScheduler_StreamChunksServer) error {
	c.streams++
	c.streamedID = req.GetJobId()
	final, err := c.chatFinalBytes()
	if err != nil {
		return err
	}
	return s.Send(&gatewaypb.ScheduleChunk{
		JobId: req.GetJobId(),
		Frame: &gatewaypb.ScheduleChunk_Completed{Completed: &gatewaypb.ScheduleCompleted{FinalResponse: final, WorkerId: "w-1"}},
	})
}

func (c *chatScheduler) GetResult(_ context.Context, _ *gatewaypb.GetResultRequest) (*gatewaypb.GetResultResponse, error) {
	final, err := c.chatFinalBytes()
	if err != nil {
		return nil, err
	}
	return &gatewaypb.GetResultResponse{Status: gatewaypb.ResultStatus_RESULT_STATUS_DONE, Body: final}, nil
}

func chatReqBody(t *testing.T, modelStr string) []byte {
	t.Helper()
	body, err := json.Marshal(chatRequest{
		Model:    modelStr,
		Messages: []chatMessage{{Role: "user", Content: "hi"}},
	})
	require.NoError(t, err)
	return body
}

// A Chat POST submits once and returns the job id (no result); the result is
// then read via GET /.v1/Jobs/{id}?wait=1, which drains the reattach stream to
// terminal and returns the decoded chat response.
func TestHandleChat_SubmitThenWaitForResult(t *testing.T) {
	h, modelStr := modelsDirWithPrimary(t)
	fake := &chatScheduler{content: "the-answer"}
	client, cleanup := newBatchChatTestClient(t, fake)
	defer cleanup()
	h.scheduler = client

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// Submit: returns the job id, does NOT block for the result.
	submitRec := httptest.NewRecorder()
	submitReq := httptest.NewRequest(http.MethodPost, "/.v1/Chat", io.NopCloser(bytes.NewReader(chatReqBody(t, modelStr)))).WithContext(ctx)
	h.handleChat(submitRec, submitReq)
	require.Equal(t, http.StatusOK, submitRec.Code, submitRec.Body.String())
	require.Equal(t, 1, fake.submits)
	require.Equal(t, "chat-job", submitRec.Header().Get(jobIDHeader))

	var sub submitResponse
	require.NoError(t, json.Unmarshal(submitRec.Body.Bytes(), &sub))
	require.Equal(t, "chat-job", sub.JobID)
	require.Equal(t, 0, fake.streams, "submit must not drain the result stream")

	// Fetch with ?wait=1: drains to terminal, returns the decoded result.
	fetchRec := httptest.NewRecorder()
	fetchReq := httptest.NewRequest(http.MethodGet, "/.v1/Jobs/chat-job?wait=1", nil).WithContext(ctx)
	fetchReq.SetPathValue("id", "chat-job")
	h.handleJobResult(fetchRec, fetchReq)
	require.Equal(t, http.StatusOK, fetchRec.Code, fetchRec.Body.String())
	require.Equal(t, 1, fake.submits, "fetch must not Submit again")
	require.Equal(t, "chat-job", fake.streamedID, "fetch re-attaches the same job")

	var out jobResultResponse
	require.NoError(t, json.Unmarshal(fetchRec.Body.Bytes(), &out))
	require.Equal(t, "done", out.Status)
	var resp chatResponse
	require.NoError(t, json.Unmarshal(out.Result, &resp))
	require.NotNil(t, resp.Message)
	require.Equal(t, "the-answer", resp.Message.Content)
}

// Build-time failures (bad JSON, invalid model path) map to 400, not 500, and
// never reach the scheduler.
func TestHandleChat_BadRequestMapsTo400(t *testing.T) {
	tests := []struct {
		name string
		body []byte
	}{
		{"malformed json", []byte("{not json")},
		{"invalid model path", func() []byte {
			b, _ := json.Marshal(chatRequest{Model: "../escape", Messages: []chatMessage{{Role: "user", Content: "hi"}}})
			return b
		}()},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h, _ := modelsDirWithPrimary(t)
			fake := &chatScheduler{content: "x"}
			client, cleanup := newBatchChatTestClient(t, fake)
			defer cleanup()
			h.scheduler = client

			rec := httptest.NewRecorder()
			req := httptest.NewRequest(http.MethodPost, "/.v1/Chat", io.NopCloser(bytes.NewReader(tt.body)))
			h.handleChat(rec, req)

			require.Equal(t, http.StatusBadRequest, rec.Code, rec.Body.String())
			require.Equal(t, 0, fake.submits, "a bad request must not reach the scheduler")
		})
	}
}
