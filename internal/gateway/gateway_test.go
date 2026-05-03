package gateway

import (
	"context"
	"net/http"
	"testing"

	gatewaypb "github.com/chinese-room-solutions/mass-proto/gen/go/gateway"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"
	"google.golang.org/grpc/metadata"
)

// fakeHandleRequestServer is a minimal gatewaypb.RuntimeGateway_HandleRequestServer
// stub that captures every Send into a slice. Recv is unused by these tests.
type fakeHandleRequestServer struct {
	grpc.ServerStream
	ctx  context.Context
	sent []*gatewaypb.HTTPResponseChunk
}

func (f *fakeHandleRequestServer) Send(msg *gatewaypb.HTTPResponseChunk) error {
	f.sent = append(f.sent, msg)
	return nil
}

func (f *fakeHandleRequestServer) Recv() (*gatewaypb.HTTPRequestChunk, error) {
	return nil, nil
}

func (f *fakeHandleRequestServer) Context() context.Context { return f.ctx }

// stub the rest of grpc.ServerStream that we don't use.
func (f *fakeHandleRequestServer) SendHeader(metadata.MD) error  { return nil }
func (f *fakeHandleRequestServer) SetHeader(metadata.MD) error   { return nil }
func (f *fakeHandleRequestServer) SetTrailer(metadata.MD)        {}
func (f *fakeHandleRequestServer) SendMsg(any) error             { return nil }
func (f *fakeHandleRequestServer) RecvMsg(any) error             { return nil }

// Tests trailer round-trip through streamResponseWriter for both net/http
// trailer conventions (pre-declared via "Trailer:" header, late-bound via
// http.TrailerPrefix). This is the foundation that gRPC piggy-backs on.
func TestStreamResponseWriter_Trailers(t *testing.T) {
	tests := []struct {
		name     string
		write    func(http.ResponseWriter)
		wantHdrs map[string]string
		wantBody []byte
		wantTrls map[string]string
	}{
		{
			name: "no trailers",
			write: func(w http.ResponseWriter) {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				_, _ = w.Write([]byte(`{"ok":true}`))
			},
			wantHdrs: map[string]string{"Content-Type": "application/json"},
			wantBody: []byte(`{"ok":true}`),
			wantTrls: nil,
		},
		{
			name: "late-bound trailer via TrailerPrefix",
			write: func(w http.ResponseWriter) {
				w.Header().Set("Content-Type", "application/grpc")
				w.WriteHeader(http.StatusOK)
				_, _ = w.Write([]byte("frame"))
				w.Header().Set(http.TrailerPrefix+"Grpc-Status", "0")
				w.Header().Set(http.TrailerPrefix+"Grpc-Message", "")
			},
			wantHdrs: map[string]string{"Content-Type": "application/grpc"},
			wantBody: []byte("frame"),
			wantTrls: map[string]string{"Grpc-Status": "0", "Grpc-Message": ""},
		},
		{
			name: "pre-declared trailer (excluded from header frame)",
			write: func(w http.ResponseWriter) {
				w.Header().Set("Content-Type", "application/grpc")
				w.Header().Set("Trailer", "Grpc-Status, Grpc-Message")
				w.WriteHeader(http.StatusOK)
				_, _ = w.Write([]byte("frame"))
				w.Header().Set("Grpc-Status", "0")
				w.Header().Set("Grpc-Message", "ok")
			},
			wantHdrs: map[string]string{
				"Content-Type": "application/grpc",
				"Trailer":      "Grpc-Status, Grpc-Message",
			},
			wantBody: []byte("frame"),
			wantTrls: map[string]string{"Grpc-Status": "0", "Grpc-Message": "ok"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := &fakeHandleRequestServer{ctx: context.Background()}
			rw := newStreamResponseWriter(srv)

			tt.write(rw)
			require.NoError(t, rw.Finish())

			require.GreaterOrEqual(t, len(srv.sent), 2, "expected at least header + EOS frames")

			// First frame: status + headers, no body, no trailers, not EOS.
			head := srv.sent[0]
			require.Equal(t, int32(http.StatusOK), head.GetStatus())
			require.Equal(t, tt.wantHdrs, head.GetHeaders())
			require.Empty(t, head.GetBody())
			require.False(t, head.GetEndOfStream())
			require.Empty(t, head.GetTrailers())

			// Last frame: EOS + trailers, no body.
			eos := srv.sent[len(srv.sent)-1]
			require.True(t, eos.GetEndOfStream())
			require.Equal(t, tt.wantTrls, eos.GetTrailers())

			// Body chunks (everything between head and eos).
			var body []byte
			for _, c := range srv.sent[1 : len(srv.sent)-1] {
				body = append(body, c.GetBody()...)
			}
			require.Equal(t, tt.wantBody, body)
		})
	}
}
