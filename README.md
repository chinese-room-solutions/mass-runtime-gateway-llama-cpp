# mass-runtime-gateway-llama-cpp

[![CI](https://github.com/chinese-room-solutions/mass-runtime-gateway-llama-cpp/actions/workflows/ci.yml/badge.svg)](https://github.com/chinese-room-solutions/mass-runtime-gateway-llama-cpp/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/chinese-room-solutions/mass-runtime-gateway-llama-cpp)](https://github.com/chinese-room-solutions/mass-runtime-gateway-llama-cpp/releases/latest)
[![Go Reference](https://pkg.go.dev/badge/github.com/chinese-room-solutions/mass-runtime-gateway-llama-cpp.svg)](https://pkg.go.dev/github.com/chinese-room-solutions/mass-runtime-gateway-llama-cpp)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

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

## Typed API (submit / poll)

Every typed endpoint is submit-then-fetch: a `POST` only *enqueues* the job
and returns `{"job_id": "..."}` (also in the `X-Mass-Job-Id` header) the
instant it is scheduled. The result is read separately from MASS's durable
result store — retained for MASS's configured result TTL, so you can poll
long after the job completes and across client reconnects. Dropping the read
connection never cancels a job; the only ways a job stops are dropping the
submit before it's scheduled, or an explicit `DELETE`.

| Endpoint | Description |
|----------|-------------|
| `POST /.v1/Chat` | Submit a chat job; returns `{"job_id": "..."}`. |
| `POST /.v1/BatchChat` | Submit a batch-chat job (one durable job for all items). |
| `POST /.v1/Embed` | Submit an embedding job. |
| `POST /.v1/BatchEmbed` | Submit a batch-embedding job. |
| `POST /.v1/Tokenize` | Submit a tokenize job. |
| `GET /.v1/Jobs/{id}` | Read any job: `{"status": "pending\|processing\|done\|error", "result": {...}, "error": "..."}`. `result` is set only when `done` and carries the shape of the job's type. Add `?wait=1` to block until the job is terminal. |
| `DELETE /.v1/Jobs/{id}` | Cancel the job, whether still queued or already running. 204 on success, 404 if it's already finished or unknown. |
| `GET /.v1/Models` | The gateway's model catalogue (from MASS's store). |

Paths are reached through MASS's proxy, e.g. `POST http://localhost:3455/mass.llama-cpp.v1/Chat`. The poll endpoint serves every job type — the stored result self-describes. Streaming has no submit/poll shape: token streaming is served by the OpenAI shim (`"stream": true`), which keeps the connection open for the duration of the job.

Batch jobs (`BatchChat`, `BatchEmbed`) are submitted at low priority: batch work is throughput-oriented and must not delay interactive chat on the same worker queue.

## Build

```bash
make build           # builds bin/mass-runtime-gateway-llama-cpp[.exe]
make package         # produces dist/mass-runtime-gateway-llama-cpp.mass for install
make test            # unit tests
```

## Install into MASS

In the MASS dashboard, **Runtimes → Install**, and pick (or drop in)
`dist/mass-runtime-gateway-llama-cpp.mass`.

After install MASS will launch the gateway on demand whenever a request lands
at `/mass.llama-cpp.v1/...` (typed API) or `/mass.llama-cpp/v1/...` (the
OpenAI-compatible shim, e.g. `/mass.llama-cpp/v1/chat/completions`).

## License

[Apache-2.0](LICENSE)
