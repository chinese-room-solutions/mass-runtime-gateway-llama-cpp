package gateway

import (
	"net/http"

	"github.com/chinese-room-solutions/mass-runtime-llama-cpp/internal/sched"
	"github.com/rs/zerolog"
)

// routerDeps groups the things every handler needs. Built once during Init.
type routerDeps struct {
	params    PluginParams
	modelsDir string
	scheduler *sched.Client
	logger    zerolog.Logger
}

// newRouter builds the http.ServeMux that handles every request MASS proxies
// in. Two surface areas:
//
//	/mass.llama-cpp.v1/Chat              POST  — typed JSON
//	/mass.llama-cpp.v1/ChatStream        POST  — typed SSE stream
//	/mass.llama-cpp.v1/BatchChat         POST
//	/mass.llama-cpp.v1/Embed             POST
//	/mass.llama-cpp.v1/BatchEmbed        POST
//	/mass.llama-cpp.v1/Tokenize          POST
//	/mass.llama-cpp.v1/LoadModel         POST
//	/mass.llama-cpp.v1/Models            GET   — gateway's view (from MASS store)
//	/mass.llama-cpp.v1/Models/Download   POST  — HuggingFace pull
//
//	/v1/chat/completions                 POST  — OpenAI-compat (stream toggle in body)
//	/v1/embeddings                       POST  — OpenAI-compat
//	/v1/models                           GET   — OpenAI-compat
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
	mux.HandleFunc("POST /.v1/ChatStream", h.handleChatStream)
	mux.HandleFunc("POST /.v1/BatchChat", h.handleBatchChat)
	mux.HandleFunc("POST /.v1/Embed", h.handleEmbed)
	mux.HandleFunc("POST /.v1/BatchEmbed", h.handleBatchEmbed)
	mux.HandleFunc("POST /.v1/Tokenize", h.handleTokenize)
	mux.HandleFunc("POST /.v1/LoadModel", h.handleLoadModel)
	mux.HandleFunc("GET /.v1/Models", h.handleListModels)
	mux.HandleFunc("POST /.v1/Models/Download", h.handleDownload)

	// OpenAI-compat shim. Same endpoints as openai-python; stream is
	// toggled via the `"stream": true` flag in the body.
	mux.HandleFunc("POST /v1/chat/completions", h.handleOpenAIChat)
	mux.HandleFunc("POST /v1/embeddings", h.handleOpenAIEmbed)
	mux.HandleFunc("GET /v1/models", h.handleOpenAIModels)

	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "not found: "+r.URL.Path, http.StatusNotFound)
	})
	return mux
}
