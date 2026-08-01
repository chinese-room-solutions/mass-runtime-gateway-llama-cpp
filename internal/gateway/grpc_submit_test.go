package gateway

import (
	"context"
	"testing"
	"time"

	llamacppv1 "github.com/chinese-room-solutions/mass-runtime-gateway-llama-cpp/gen/go/llama_cpp/v1"
	"github.com/stretchr/testify/require"
)

// The gRPC surface mirrors the HTTP submit/fetch split: SubmitChat enqueues and
// returns a job id; GetResult(wait=true) drains to terminal and returns the
// typed result via the JobResult oneof.
func TestGRPC_SubmitChatThenGetResult(t *testing.T) {
	h, modelStr := modelsDirWithPrimary(t)
	fake := &chatScheduler{content: "the-answer"}
	client, cleanup := newBatchChatTestClient(t, fake)
	defer cleanup()
	h.scheduler = client
	s := newGRPCServer(h)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	sub, err := s.SubmitChat(ctx, &llamacppv1.ChatRequest{
		Model:    modelStr,
		Messages: []*llamacppv1.Message{{Role: llamacppv1.Role_ROLE_USER, Content: "hi"}},
	})
	require.NoError(t, err)
	require.Equal(t, "chat-job", sub.GetJobId())
	require.Equal(t, 1, fake.submits)
	require.Equal(t, 0, fake.streams, "submit must not drain the result stream")

	res, err := s.GetResult(ctx, &llamacppv1.GetResultRequest{JobId: sub.GetJobId(), Wait: true})
	require.NoError(t, err)
	require.Equal(t, llamacppv1.JobStatus_JOB_STATUS_DONE, res.GetStatus())
	require.Equal(t, 1, fake.submits, "fetch must not Submit again")
	require.Equal(t, "chat-job", fake.streamedID, "fetch re-attaches the same job")

	chat := res.GetChat()
	require.NotNil(t, chat)
	require.NotNil(t, chat.GetMessage())
	require.Equal(t, "the-answer", chat.GetMessage().GetContent())
}

func TestGRPC_SubmitChat_MissingModel(t *testing.T) {
	h, _ := modelsDirWithPrimary(t)
	fake := &chatScheduler{}
	client, cleanup := newBatchChatTestClient(t, fake)
	defer cleanup()
	h.scheduler = client
	s := newGRPCServer(h)

	_, err := s.SubmitChat(context.Background(), &llamacppv1.ChatRequest{})
	require.Error(t, err)
	require.Equal(t, 0, fake.submits, "invalid request must not reach the scheduler")
}

func TestGRPC_GetResult_MissingID(t *testing.T) {
	h, _ := modelsDirWithPrimary(t)
	s := newGRPCServer(h)
	_, err := s.GetResult(context.Background(), &llamacppv1.GetResultRequest{})
	require.Error(t, err)
}
