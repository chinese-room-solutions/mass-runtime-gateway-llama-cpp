# mass-runtime-llama-cpp

Runtime gateway for the **llama.cpp** inference runtime, packaged as a
[MASS](https://github.com/chinese-room-solutions/mass) `.mass` package.

## What it does

This binary is a [hashicorp/go-plugin](https://github.com/hashicorp/go-plugin)
subprocess MASS launches once installed. It owns:

- The HTTP API at `/mass.llama-cpp.v1/*` (typed Chat / Embed / Tokenize) and the
  OpenAI-compatible shim at `/v1/chat/completions`, `/v1/embeddings`,
  `/v1/models`.
- GGUF metadata parsing for the Models tab.
- HuggingFace downloads + local-path imports of GGUF models.
- The gateway↔worker payload encoding (proto in
  [`mass-proto/proto/llama-cpp/payload.proto`](https://github.com/chinese-room-solutions/mass-proto)).

The actual inference happens on **mass-worker-llama-cpp** workers — this gateway
forwards opaque payloads via MASS's scheduler.

## Build

```bash
make build           # builds bin/mass-runtime-llama-cpp[.exe]
make package         # produces dist/mass-runtime-llama-cpp.mass for install
```

## Install into MASS

Drop `dist/mass-runtime-llama-cpp.mass` onto MASS's Runtimes tab, or:

```bash
curl -X POST http://localhost:3455/api/runtimes/install \
  -H 'Content-Type: application/json' \
  -d '{"package_path":"/abs/path/to/mass-runtime-llama-cpp.mass"}'
```

After install MASS will launch the gateway on demand whenever a request lands
at `/mass.llama-cpp.v1/...` or `/v1/chat/completions`.
