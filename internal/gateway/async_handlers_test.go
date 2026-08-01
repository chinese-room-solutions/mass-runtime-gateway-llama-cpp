package gateway

import (
	"context"
	"encoding/json"
	"errors"
	"net"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	gatewaypb "github.com/chinese-room-solutions/mass-proto/gen/go/gateway"
	llamacpp "github.com/chinese-room-solutions/mass-proto/gen/go/llama-cpp"
	"github.com/chinese-room-solutions/mass-runtime-gateway-llama-cpp/internal/payload"
	"github.com/chinese-room-solutions/mass-runtime-gateway-llama-cpp/internal/sched"
	"github.com/rs/zerolog"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
	"google.golang.org/grpc/test/bufconn"
)

// asyncFakeScheduler serves GetResult / CancelJob from injected hooks.
type asyncFakeScheduler struct {
	gatewaypb.UnimplementedMassSchedulerServer
	getResult func(*gatewaypb.GetResultRequest) (*gatewaypb.GetResultResponse, error)
	cancel    func(*gatewaypb.CancelJobRequest) (*gatewaypb.CancelJobResponse, error)
}

func (f *asyncFakeScheduler) GetResult(_ context.Context, r *gatewaypb.GetResultRequest) (*gatewaypb.GetResultResponse, error) {
	return f.getResult(r)
}
func (f *asyncFakeScheduler) CancelJob(_ context.Context, r *gatewaypb.CancelJobRequest) (*gatewaypb.CancelJobResponse, error) {
	return f.cancel(r)
}

// newAsyncTestHandlers builds a handlers wired to a bufconn-backed sched.Client
// over the given fake, plus a ServeMux mounting the async routes so PathValue
// ({id}) resolves exactly as in production.
func newAsyncTestHandlers(t *testing.T, fake *asyncFakeScheduler) (*http.ServeMux, func()) {
	t.Helper()
	lis := bufconn.Listen(1 << 16)
	grpcServer := grpc.NewServer()
	gatewaypb.RegisterMassSchedulerServer(grpcServer, fake)
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

	h := &handlers{scheduler: sched.NewClient(conn, zerolog.Nop()), logger: zerolog.Nop()}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /.v1/Jobs/{id}", h.handleJobResult)
	mux.HandleFunc("DELETE /.v1/Jobs/{id}", h.handleJobCancel)

	cleanup := func() { _ = conn.Close(); grpcServer.Stop(); _ = lis.Close() }
	return mux, cleanup
}

// doneBody encodes a ChatFinal JobChunk the way a worker terminal frame does,
// so handleJobResult's decode path is exercised end-to-end.
func doneBody(t *testing.T, content string) []byte {
	t.Helper()
	b, err := payload.EncodeJobChunk(&llamacpp.JobChunk{
		Body: &llamacpp.JobChunk_ChatFinal{ChatFinal: &llamacpp.ChatFinal{
			Message:      &llamacpp.ChatMessage{Role: llamacpp.Role_ROLE_ASSISTANT, Content: content},
			FinishReason: llamacpp.FinishReason_FINISH_REASON_STOP,
		}},
	})
	require.NoError(t, err)
	return b
}

func TestHandleJobResult(t *testing.T) {
	tests := []struct {
		name        string
		resp        *gatewaypb.GetResultResponse
		grpcErr     error
		wantStatus  int
		wantBodyHas string
	}{
		{
			name:        "pending",
			resp:        &gatewaypb.GetResultResponse{Status: gatewaypb.ResultStatus_RESULT_STATUS_PENDING},
			wantStatus:  http.StatusOK,
			wantBodyHas: `"status":"pending"`,
		},
		{
			name:        "processing",
			resp:        &gatewaypb.GetResultResponse{Status: gatewaypb.ResultStatus_RESULT_STATUS_PROCESSING},
			wantStatus:  http.StatusOK,
			wantBodyHas: `"status":"processing"`,
		},
		{
			name:        "error",
			resp:        &gatewaypb.GetResultResponse{Status: gatewaypb.ResultStatus_RESULT_STATUS_ERROR, Error: "boom"},
			wantStatus:  http.StatusOK,
			wantBodyHas: `"error":"boom"`,
		},
		{
			name:       "not found -> 404",
			grpcErr:    status.Error(codes.NotFound, "gone"),
			wantStatus: http.StatusNotFound,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fake := &asyncFakeScheduler{getResult: func(*gatewaypb.GetResultRequest) (*gatewaypb.GetResultResponse, error) {
				return tt.resp, tt.grpcErr
			}}
			mux, cleanup := newAsyncTestHandlers(t, fake)
			defer cleanup()

			rec := httptest.NewRecorder()
			mux.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/.v1/Jobs/rid-1", nil))

			require.Equal(t, tt.wantStatus, rec.Code)
			if tt.wantBodyHas != "" {
				require.Contains(t, rec.Body.String(), tt.wantBodyHas)
			}
		})
	}
}

// pollDone runs a DONE poll with the given stored body and returns the decoded
// jobResultResponse.
func pollDone(t *testing.T, body []byte) jobResultResponse {
	t.Helper()
	fake := &asyncFakeScheduler{getResult: func(*gatewaypb.GetResultRequest) (*gatewaypb.GetResultResponse, error) {
		return &gatewaypb.GetResultResponse{Status: gatewaypb.ResultStatus_RESULT_STATUS_DONE, Body: body}, nil
	}}
	mux, cleanup := newAsyncTestHandlers(t, fake)
	defer cleanup()

	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/.v1/Jobs/rid-1", nil))
	require.Equal(t, http.StatusOK, rec.Code)

	var got jobResultResponse
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &got))
	require.Equal(t, "done", got.Status)
	return got
}

func TestHandleJobResult_DoneDecodesPerType(t *testing.T) {
	t.Run("chat", func(t *testing.T) {
		got := pollDone(t, doneBody(t, "the-answer"))
		var resp chatResponse
		require.NoError(t, json.Unmarshal(got.Result, &resp))
		require.NotNil(t, resp.Message)
		require.Equal(t, "the-answer", resp.Message.Content)
	})

	t.Run("embed", func(t *testing.T) {
		body, err := payload.EncodeJobChunk(&llamacpp.JobChunk{
			Body: &llamacpp.JobChunk_Embed{Embed: &llamacpp.EmbedResult{Embedding: []float32{1, 2, 3}}},
		})
		require.NoError(t, err)
		got := pollDone(t, body)
		var resp embedResponse
		require.NoError(t, json.Unmarshal(got.Result, &resp))
		require.Equal(t, []float32{1, 2, 3}, resp.Embedding)
	})

	t.Run("tokenize", func(t *testing.T) {
		body, err := payload.EncodeJobChunk(&llamacpp.JobChunk{
			Body: &llamacpp.JobChunk_Tokenize{Tokenize: &llamacpp.TokenizeResult{Tokens: []int32{5, 6, 7}}},
		})
		require.NoError(t, err)
		got := pollDone(t, body)
		var resp tokenizeResponse
		require.NoError(t, json.Unmarshal(got.Result, &resp))
		require.Equal(t, []int32{5, 6, 7}, resp.Tokens)
	})

	t.Run("batch_chat", func(t *testing.T) {
		body, err := payload.EncodeJobChunk(&llamacpp.JobChunk{
			Body: &llamacpp.JobChunk_BatchChat{BatchChat: &llamacpp.BatchChatResult{
				Id: "bchat-1",
				Items: []*llamacpp.ChatFinal{
					{Id: "i0", Message: &llamacpp.ChatMessage{Role: llamacpp.Role_ROLE_ASSISTANT, Content: "first"}, FinishReason: llamacpp.FinishReason_FINISH_REASON_STOP},
					{Id: "i1", Message: &llamacpp.ChatMessage{Role: llamacpp.Role_ROLE_ASSISTANT, Content: "second"}, FinishReason: llamacpp.FinishReason_FINISH_REASON_STOP},
				},
			}},
		})
		require.NoError(t, err)
		got := pollDone(t, body)
		var resp batchChatResponse
		require.NoError(t, json.Unmarshal(got.Result, &resp))
		require.Len(t, resp.Responses, 2)
		require.Equal(t, "first", resp.Responses[0].Message.Content)
		require.Equal(t, "second", resp.Responses[1].Message.Content)
		require.Equal(t, "i1", resp.Responses[1].ID)
	})
}

func TestHandleJobCancel(t *testing.T) {
	t.Run("success -> 204", func(t *testing.T) {
		fake := &asyncFakeScheduler{cancel: func(*gatewaypb.CancelJobRequest) (*gatewaypb.CancelJobResponse, error) {
			return &gatewaypb.CancelJobResponse{}, nil
		}}
		mux, cleanup := newAsyncTestHandlers(t, fake)
		defer cleanup()
		rec := httptest.NewRecorder()
		mux.ServeHTTP(rec, httptest.NewRequest(http.MethodDelete, "/.v1/Jobs/rid-1", nil))
		require.Equal(t, http.StatusNoContent, rec.Code)
	})

	t.Run("not found -> 404", func(t *testing.T) {
		fake := &asyncFakeScheduler{cancel: func(*gatewaypb.CancelJobRequest) (*gatewaypb.CancelJobResponse, error) {
			return nil, status.Error(codes.NotFound, "gone")
		}}
		mux, cleanup := newAsyncTestHandlers(t, fake)
		defer cleanup()
		rec := httptest.NewRecorder()
		mux.ServeHTTP(rec, httptest.NewRequest(http.MethodDelete, "/.v1/Jobs/rid-1", nil))
		require.Equal(t, http.StatusNotFound, rec.Code)
	})
}
