package sched

import (
	"context"
	"testing"

	gatewaypb "github.com/chinese-room-solutions/mass-proto/gen/go/gateway"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

func TestClient_GetResult(t *testing.T) {
	tests := []struct {
		name       string
		resp       *gatewaypb.GetResultResponse
		grpcErr    error
		wantErrIs  error
		wantStatus ResultStatus
		wantBody   string
		wantErrTxt string
	}{
		{
			name:       "done with body",
			resp:       &gatewaypb.GetResultResponse{Status: gatewaypb.ResultStatus_RESULT_STATUS_DONE, Body: []byte("hi")},
			wantStatus: ResultDone,
			wantBody:   "hi",
		},
		{
			name:       "processing, empty body",
			resp:       &gatewaypb.GetResultResponse{Status: gatewaypb.ResultStatus_RESULT_STATUS_PROCESSING},
			wantStatus: ResultProcessing,
		},
		{
			name:       "error carries message",
			resp:       &gatewaypb.GetResultResponse{Status: gatewaypb.ResultStatus_RESULT_STATUS_ERROR, Error: "boom"},
			wantStatus: ResultError,
			wantErrTxt: "boom",
		},
		{
			name:      "not found maps to ErrResultNotFound",
			grpcErr:   status.Error(codes.NotFound, "unknown"),
			wantErrIs: ErrResultNotFound,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fake := &fakeMassScheduler{
				getResultFn: func(_ *gatewaypb.GetResultRequest) (*gatewaypb.GetResultResponse, error) {
					return tt.resp, tt.grpcErr
				},
			}
			client, cleanup := newTestClient(t, fake)
			defer cleanup()

			res, err := client.GetResult(context.Background(), "rid")
			if tt.wantErrIs != nil {
				require.ErrorIs(t, err, tt.wantErrIs)
				return
			}
			require.NoError(t, err)
			require.Equal(t, tt.wantStatus, res.Status)
			require.Equal(t, tt.wantBody, string(res.Body))
			require.Equal(t, tt.wantErrTxt, res.Err)
		})
	}
}

func TestClient_CancelJob(t *testing.T) {
	t.Run("success", func(t *testing.T) {
		fake := &fakeMassScheduler{
			cancelJobFn: func(_ *gatewaypb.CancelJobRequest) (*gatewaypb.CancelJobResponse, error) {
				return &gatewaypb.CancelJobResponse{}, nil
			},
		}
		client, cleanup := newTestClient(t, fake)
		defer cleanup()
		require.NoError(t, client.CancelJob(context.Background(), "rid"))
	})

	t.Run("not found maps to ErrResultNotFound", func(t *testing.T) {
		fake := &fakeMassScheduler{
			cancelJobFn: func(_ *gatewaypb.CancelJobRequest) (*gatewaypb.CancelJobResponse, error) {
				return nil, status.Error(codes.NotFound, "gone")
			},
		}
		client, cleanup := newTestClient(t, fake)
		defer cleanup()
		require.ErrorIs(t, client.CancelJob(context.Background(), "rid"), ErrResultNotFound)
	})
}

func TestClient_SubmitOnlyReturnsJobID(t *testing.T) {
	fake := &fakeMassScheduler{} // Submit returns "job-1"
	client, cleanup := newTestClient(t, fake)
	defer cleanup()

	id, err := client.SubmitOnly(context.Background(), ScheduleParams{ModelID: "m", Cost: 1})
	require.NoError(t, err)
	require.Equal(t, "job-1", id)
}
