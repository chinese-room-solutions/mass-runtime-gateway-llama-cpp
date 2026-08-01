package gateway

import (
	"context"
	"net/http"
	"strings"

	"google.golang.org/grpc/metadata"
)

// sourceContextKey identifies the inbound request's caller in ctx. Populated
// at the entry point (HTTP middleware or gRPC interceptor) so deeper
// helpers (Schedule submitting envelopes with source attribution) can pick
// it up without rewiring every call site.
type sourceContextKey struct{}

// withSource attaches a caller identity to ctx. Empty source is a no-op.
func withSource(ctx context.Context, source string) context.Context {
	source = strings.TrimSpace(source)
	if source == "" {
		return ctx
	}
	return context.WithValue(ctx, sourceContextKey{}, source)
}

// sourceFromContext returns the caller identity stored on ctx. Looks at
// both the context-value (set by httpSourceMiddleware) and inbound gRPC
// metadata (set by clients via metadata.New("x-mass-source")). Defaults to
// "direct" — the convention MASS used pre-refactor for raw requests with
// no app attribution.
func sourceFromContext(ctx context.Context) string {
	if v, ok := ctx.Value(sourceContextKey{}).(string); ok && v != "" {
		return v
	}
	if v := grpcSourceFromMetadata(ctx); v != "" {
		return v
	}
	return "direct"
}

// httpSourceMiddleware extracts X-Mass-Source from the inbound request and
// stashes it on ctx so downstream handlers (typed JSON, OpenAI, typed gRPC
// proxied as HTTP) can pick it up via sourceFromContext. Mounts at the
// outermost layer of the http.ServeMux router so every endpoint inherits
// the value.
func httpSourceMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if v := r.Header.Get("X-Mass-Source"); v != "" {
			r = r.WithContext(withSource(r.Context(), v))
		}
		next.ServeHTTP(w, r)
	})
}

// grpcSourceFromMetadata reads x-mass-source from inbound gRPC metadata
// (headers downcase). gRPC clients send the same identity via
// `metadata.New(map[string]string{"x-mass-source": "..."})`.
func grpcSourceFromMetadata(ctx context.Context) string {
	md, ok := metadata.FromIncomingContext(ctx)
	if !ok {
		return ""
	}
	for _, v := range md.Get("x-mass-source") {
		v = strings.TrimSpace(v)
		if v != "" {
			return v
		}
	}
	return ""
}
