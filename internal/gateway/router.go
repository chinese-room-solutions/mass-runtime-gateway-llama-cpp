package gateway

import (
	"net/http"

	"github.com/chinese-room-solutions/mass-runtime-gateway-llama-cpp/internal/sched"
	"github.com/chinese-room-solutions/mass-sdk/uikit"
	"github.com/rs/zerolog"
)

// routerDeps groups the things every handler needs. Built once during Init.
type routerDeps struct {
	params      PluginParams
	runtimeName string
	modelsDir   string
	scheduler   *sched.Client
	cache       *metadataCache
	logger      zerolog.Logger
}

// newRouter builds the http.ServeMux that handles every request MASS proxies
// in. Surface areas:
//
//	/mass.llama-cpp.v1/Chat              POST  — submit, returns {job_id}
//	/mass.llama-cpp.v1/BatchChat         POST  — submit
//	/mass.llama-cpp.v1/Embed             POST  — submit
//	/mass.llama-cpp.v1/BatchEmbed        POST  — submit
//	/mass.llama-cpp.v1/Tokenize          POST  — submit
//	/mass.llama-cpp.v1/Jobs/{id}         GET    — read result (?wait=1 blocks)
//	                                     DELETE — cancel a job
//	/mass.llama-cpp.v1/Models            GET   — gateway's view (from MASS store)
//
//	/v1/chat/completions                 POST  — OpenAI-compat (stream toggle in body)
//	/v1/embeddings                       POST  — OpenAI-compat
//	/v1/models                           GET   — OpenAI-compat
//
// Submit/fetch split. A typed POST only *submits*: it enqueues the job and
// returns {"job_id":"..."} the instant it's scheduled (also in the
// X-Mass-Job-Id header). Pre-schedule failures carry an honest HTTP 4xx/5xx.
// The result is then read via GET Jobs/{id}: ?wait=1 blocks until the job is
// terminal (done|error), otherwise it returns the current status (poll). That
// read is durable and side-effect-free — dropping it never cancels; reconnect
// or poll later. The ONLY ways a job stops are dropping the Submit before it's
// scheduled, or DELETE Jobs/{id}. Execution failures surface as the result's
// own error status, not an HTTP code.
//
// MASS strips the `/mass.llama-cpp` prefix before forwarding, so paths
// arrive here as `.v1/...`. The leading dot survives the split — see
// internal/web/handler.go::handleRuntimeProxy on the MASS side.
func newRouter(d routerDeps) http.Handler {
	mux := http.NewServeMux()
	h := newHandlers(d)

	// Typed API. The leading "." is from MASS's path split (it forwards
	// everything after the runtime kind, including the dot separator).
	mux.HandleFunc("POST /.v1/Chat", h.handleChat)
	mux.HandleFunc("POST /.v1/BatchChat", h.handleBatchChat)
	mux.HandleFunc("POST /.v1/Embed", h.handleEmbed)
	mux.HandleFunc("POST /.v1/BatchEmbed", h.handleBatchEmbed)
	mux.HandleFunc("POST /.v1/Tokenize", h.handleTokenize)
	mux.HandleFunc("GET /.v1/Jobs/{id}", h.handleJobResult)
	mux.HandleFunc("DELETE /.v1/Jobs/{id}", h.handleJobCancel)
	mux.HandleFunc("GET /.v1/Models", h.handleListModels)
	mux.HandleFunc("GET /.v1/Models/Detail", h.handleModelDetailHTML)

	// OpenAI-compat shim. Same endpoints as openai-python; stream is
	// toggled via the `"stream": true` flag in the body.
	mux.HandleFunc("POST /v1/chat/completions", h.handleOpenAIChat)
	mux.HandleFunc("POST /v1/embeddings", h.handleOpenAIEmbed)
	mux.HandleFunc("GET /v1/models", h.handleOpenAIModels)

	// Operator-facing install UI. MASS mounts /mass.<runtime>.*/install
	// in its iframe modal; the page hosts the HF picker and posts the
	// resolved file list back via /.install/submit, which calls
	// MassScheduler.DownloadFiles for the actual fetch.
	installMux := installRouter(d)
	mux.Handle("GET /.install", installMux)
	mux.Handle("POST /.install/search", installMux)
	mux.Handle("POST /.install/more", installMux)
	mux.Handle("POST /.install/submit", installMux)

	// Front-end assets (Shoelace, Datastar) for the install page. The page
	// references them under /mass.<runtime>/_uikit/... (LayoutUnder); MASS's
	// proxy strips the /mass.<runtime> prefix, so they arrive here at the
	// mount's own /_uikit/... path.
	uikit.MountAssets(mux)

	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "not found: "+r.URL.Path, http.StatusNotFound)
	})
	// Wrap the mux so every handler sees X-Mass-Source on its ctx.
	return httpSourceMiddleware(mux)
}
