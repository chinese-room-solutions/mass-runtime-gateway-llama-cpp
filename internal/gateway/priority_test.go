package gateway

import (
	"context"
	"testing"
	"time"

	gatewaypb "github.com/chinese-room-solutions/mass-proto/gen/go/gateway"
	llamacpp "github.com/chinese-room-solutions/mass-proto/gen/go/llama-cpp"
	"github.com/rs/zerolog"
	"github.com/stretchr/testify/require"
)

// Batch work is throughput-oriented; the gateway marks it LOW so MASS
// dequeues interactive jobs first within the same worker queue. Everything
// else stays UNSPECIFIED (MEDIUM on the MASS side).
func TestBuildScheduleParamsBatchPriority(t *testing.T) {
	h := &handlers{logger: zerolog.Nop()}
	tests := []struct {
		name string
		job  *llamacpp.Job
		want gatewaypb.JobPriority
	}{
		{
			name: "chat stays unspecified",
			job: &llamacpp.Job{
				Kind: llamacpp.JobKind_JOB_KIND_CHAT,
				Body: &llamacpp.Job_Chat{Chat: &llamacpp.ChatJob{}},
			},
			want: gatewaypb.JobPriority_JOB_PRIORITY_UNSPECIFIED,
		},
		{
			name: "embed stays unspecified",
			job: &llamacpp.Job{
				Kind: llamacpp.JobKind_JOB_KIND_EMBED,
				Body: &llamacpp.Job_Embed{Embed: &llamacpp.EmbedJob{Input: "x"}},
			},
			want: gatewaypb.JobPriority_JOB_PRIORITY_UNSPECIFIED,
		},
		{
			name: "batch chat is low",
			job: &llamacpp.Job{
				Kind: llamacpp.JobKind_JOB_KIND_BATCH_CHAT,
				Body: &llamacpp.Job_BatchChat{BatchChat: &llamacpp.BatchChatJob{}},
			},
			want: gatewaypb.JobPriority_JOB_PRIORITY_LOW,
		},
		{
			name: "batch embed is low",
			job: &llamacpp.Job{
				Kind: llamacpp.JobKind_JOB_KIND_BATCH_EMBED,
				Body: &llamacpp.Job_BatchEmbed{BatchEmbed: &llamacpp.BatchEmbedJob{Inputs: []string{"x"}}},
			},
			want: gatewaypb.JobPriority_JOB_PRIORITY_LOW,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p, err := h.buildScheduleParams(context.Background(), "model-1", tt.job, nil, nil)
			require.NoError(t, err)
			require.Equal(t, tt.want, p.Priority)
		})
	}
}

// The priority set in ScheduleParams must survive the sched client's
// translation onto the SubmitRequest wire shape.
func TestDispatchPassesBatchPriorityToSubmit(t *testing.T) {
	job := &llamacpp.Job{
		Kind: llamacpp.JobKind_JOB_KIND_BATCH_EMBED,
		Body: &llamacpp.Job_BatchEmbed{BatchEmbed: &llamacpp.BatchEmbedJob{Inputs: []string{"x"}}},
	}

	fake := &captureScheduler{}
	client, cleanup := newDispatchTestClient(t, fake)
	defer cleanup()

	h := &handlers{scheduler: client, logger: zerolog.Nop()}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	_, chunks, err := h.dispatchWithID(ctx, "model-1", job, nil, nil)
	require.NoError(t, err)
	for range chunks {
	}

	require.Equal(t, gatewaypb.JobPriority_JOB_PRIORITY_LOW, fake.lastPriority.Load().(gatewaypb.JobPriority))
}
