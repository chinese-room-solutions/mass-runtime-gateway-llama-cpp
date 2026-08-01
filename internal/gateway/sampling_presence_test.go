package gateway

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	llamacpp "github.com/chinese-room-solutions/mass-proto/gen/go/llama-cpp"
	llamacppv1 "github.com/chinese-room-solutions/mass-runtime-gateway-llama-cpp/gen/go/llama_cpp/v1"
	"github.com/chinese-room-solutions/mass-runtime-gateway-llama-cpp/internal/payload"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"
)

// submittedSampling decodes the job the fake scheduler captured and returns
// its chat sampling params — exactly what would hit the worker's wire.
func submittedSampling(t *testing.T, fake *chatScheduler) *llamacpp.SamplingParams {
	t.Helper()
	require.NotEmpty(t, fake.lastPayload, "handler must have submitted a job")
	job, err := payload.DecodeJob(fake.lastPayload)
	require.NoError(t, err)
	require.NotNil(t, job.GetChat())
	return job.GetChat().GetSampling()
}

// Presence must survive from client JSON to the wire: omitted fields stay
// absent (worker default applies), explicit zeros arrive present (temperature
// 0 → greedy, seed 0 → that exact seed).
func TestSamplingPresence_TypedChat(t *testing.T) {
	cases := []struct {
		name         string
		samplingJSON string
		check        func(t *testing.T, sp *llamacpp.SamplingParams)
	}{
		{
			name:         "omitted temperature and seed stay absent",
			samplingJSON: `{"max_tokens":16}`,
			check: func(t *testing.T, sp *llamacpp.SamplingParams) {
				require.NotNil(t, sp)
				require.Nil(t, sp.Temperature)
				require.Nil(t, sp.Seed)
				require.Nil(t, sp.TopP)
				require.Nil(t, sp.TopK)
				require.Nil(t, sp.MinP)
				require.Nil(t, sp.RepeatPenalty)
				require.Nil(t, sp.FrequencyPenalty)
				require.Nil(t, sp.PresencePenalty)
				require.NotNil(t, sp.MaxTokens)
				require.Equal(t, int32(16), *sp.MaxTokens)
			},
		},
		{
			name:         "explicit zeros arrive present",
			samplingJSON: `{"temperature":0,"seed":0}`,
			check: func(t *testing.T, sp *llamacpp.SamplingParams) {
				require.NotNil(t, sp)
				require.NotNil(t, sp.Temperature, "temperature:0 must be present (greedy), not absent")
				require.Equal(t, float32(0), *sp.Temperature)
				require.NotNil(t, sp.Seed, "seed:0 must be present (deterministic seed 0), not absent")
				require.Equal(t, int32(0), *sp.Seed)
				require.Nil(t, sp.MaxTokens)
			},
		},
		{
			name:         "explicit values pass through",
			samplingJSON: `{"temperature":0.7,"seed":42,"top_k":40}`,
			check: func(t *testing.T, sp *llamacpp.SamplingParams) {
				require.NotNil(t, sp)
				require.NotNil(t, sp.Temperature)
				require.InDelta(t, 0.7, *sp.Temperature, 1e-6)
				require.NotNil(t, sp.Seed)
				require.Equal(t, int32(42), *sp.Seed)
				require.NotNil(t, sp.TopK)
				require.Equal(t, int32(40), *sp.TopK)
			},
		},
	}

	h, modelStr := modelsDirWithPrimary(t)
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			fake := &chatScheduler{content: "ok"}
			client, cleanup := newBatchChatTestClient(t, fake)
			defer cleanup()
			h.scheduler = client

			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()

			body := fmt.Sprintf(`{"model":%q,"messages":[{"role":"user","content":"hi"}],"sampling":%s}`, modelStr, tc.samplingJSON)
			rec := httptest.NewRecorder()
			req := httptest.NewRequest(http.MethodPost, "/.v1/Chat", io.NopCloser(bytes.NewReader([]byte(body)))).WithContext(ctx)
			h.handleChat(rec, req)
			require.Equal(t, http.StatusOK, rec.Code, rec.Body.String())

			tc.check(t, submittedSampling(t, fake))
		})
	}
}

// The typed gRPC API carries presence too: samplingFromProto must map
// absent fields to nil and present fields (including zeros) to pointers.
func TestSamplingFromProto_Presence(t *testing.T) {
	cases := []struct {
		name  string
		in    *llamacppv1.Sampling
		check func(t *testing.T, sp *samplingParams)
	}{
		{
			name: "nil message -> nil params",
			in:   nil,
			check: func(t *testing.T, sp *samplingParams) {
				require.Nil(t, sp)
			},
		},
		{
			name: "absent fields stay absent",
			in:   &llamacppv1.Sampling{Stop: []string{"###"}},
			check: func(t *testing.T, sp *samplingParams) {
				require.NotNil(t, sp)
				require.Nil(t, sp.MaxTokens)
				require.Nil(t, sp.Temperature)
				require.Nil(t, sp.TopP)
				require.Nil(t, sp.TopK)
				require.Nil(t, sp.Seed)
				require.Nil(t, sp.MinP)
				require.Nil(t, sp.RepeatPenalty)
				require.Nil(t, sp.FrequencyPenalty)
				require.Nil(t, sp.PresencePenalty)
				require.Equal(t, []string{"###"}, sp.Stop)
			},
		},
		{
			name: "explicit zeros arrive present",
			in:   &llamacppv1.Sampling{Temperature: proto.Float32(0), Seed: proto.Int32(0)},
			check: func(t *testing.T, sp *samplingParams) {
				require.NotNil(t, sp.Temperature)
				require.Equal(t, float32(0), *sp.Temperature)
				require.NotNil(t, sp.Seed)
				require.Equal(t, int32(0), *sp.Seed)
			},
		},
		{
			name: "values pass through as copies",
			in:   &llamacppv1.Sampling{Temperature: proto.Float32(0.7), Seed: proto.Int32(42), TopK: proto.Int32(40)},
			check: func(t *testing.T, sp *samplingParams) {
				require.NotNil(t, sp.Temperature)
				require.InDelta(t, 0.7, *sp.Temperature, 1e-6)
				require.Equal(t, int32(42), *sp.Seed)
				require.Equal(t, int32(40), *sp.TopK)
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			tc.check(t, samplingFromProto(tc.in))
		})
	}
}

// The OpenAI-compatible endpoint must keep the same presence semantics.
func TestSamplingPresence_OpenAIChat(t *testing.T) {
	cases := []struct {
		name      string
		extraJSON string // appended after model+messages, "" = nothing extra
		check     func(t *testing.T, sp *llamacpp.SamplingParams)
	}{
		{
			name:      "omitted temperature and seed stay absent",
			extraJSON: `,"max_tokens":16`,
			check: func(t *testing.T, sp *llamacpp.SamplingParams) {
				require.NotNil(t, sp)
				require.Nil(t, sp.Temperature)
				require.Nil(t, sp.Seed)
				require.Nil(t, sp.TopP)
				require.Nil(t, sp.FrequencyPenalty)
				require.Nil(t, sp.PresencePenalty)
				require.NotNil(t, sp.MaxTokens)
				require.Equal(t, int32(16), *sp.MaxTokens)
			},
		},
		{
			name:      "explicit zeros arrive present",
			extraJSON: `,"temperature":0,"seed":0`,
			check: func(t *testing.T, sp *llamacpp.SamplingParams) {
				require.NotNil(t, sp)
				require.NotNil(t, sp.Temperature, "temperature:0 must be present (greedy), not absent")
				require.Equal(t, float32(0), *sp.Temperature)
				require.NotNil(t, sp.Seed, "seed:0 must be present (deterministic seed 0), not absent")
				require.Equal(t, int32(0), *sp.Seed)
			},
		},
		{
			name:      "explicit values pass through",
			extraJSON: `,"temperature":1.2,"seed":7,"top_p":0.9`,
			check: func(t *testing.T, sp *llamacpp.SamplingParams) {
				require.NotNil(t, sp)
				require.NotNil(t, sp.Temperature)
				require.InDelta(t, 1.2, *sp.Temperature, 1e-6)
				require.NotNil(t, sp.Seed)
				require.Equal(t, int32(7), *sp.Seed)
				require.NotNil(t, sp.TopP)
				require.InDelta(t, 0.9, *sp.TopP, 1e-6)
			},
		},
	}

	h, modelStr := modelsDirWithPrimary(t)
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			fake := &chatScheduler{content: "ok"}
			client, cleanup := newBatchChatTestClient(t, fake)
			defer cleanup()
			h.scheduler = client

			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()

			body := fmt.Sprintf(`{"model":%q,"messages":[{"role":"user","content":"hi"}]%s}`, modelStr, tc.extraJSON)
			rec := httptest.NewRecorder()
			req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", io.NopCloser(bytes.NewReader([]byte(body)))).WithContext(ctx)
			h.handleOpenAIChat(rec, req)
			require.Equal(t, http.StatusOK, rec.Code, rec.Body.String())

			tc.check(t, submittedSampling(t, fake))
		})
	}
}
