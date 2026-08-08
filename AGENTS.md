## Working style
- Top-level agent: plan, orchestrate, and review. Make simple changes yourself — a settled, small, contained edit costs more to hand off than to make.
- Delegate the rest, at most 2 subagents at a time: work spanning several files, needing its own exploration, or running long. Each starts cold — hand it the diagnosis, file refs, design, environment setup, and what to verify.
- Subagent: do the work yourself. Never spawn further agents.
- Scale verification to the risk of being wrong. A cosmetic markup/CSS change needs a rebuild and one look at it; a cross-engine, multi-file, or unproven diagnosis needs measured numbers. Over-specified verification is how a small fix gets expensive.
- Verify a diagnosis against current code before fixing it. One commit per fix.
- Commit messages: as short as the change allows — one subject line, a body only for a why the diff doesn't show. No `Co-Authored-By` or any other trailer.

## Code quality
- Follow the best practices of this repo's stack, and good project organization.
- Write simple, reusable, maintainable code. Maintainability and simplicity come first; seek optimizations only after that.
- Don't use dirty workarounds unless there's truly no other way.
- You can and should make any breaking changes needed.
- Avoid over-generalizing for hypothetical future use — write the minimal thing first.
- Reusable helpers belong in mass-sdk, not copy-pasted between MASS repos. When changing the SDK's public surface, update the dependent repos in the same change.

## Tidiness
- Revisit your changes: simplify what can be simplified, remove what's no longer needed, and don't leave unnecessary moves behind.
- When a fix lands after several attempts, revisit the trail before committing: re-verify each accumulated change is load-bearing (would the issue return without it?) and revert the ones that aren't — ship only the code that actually fixes the issue.
- Keep comments and docs concise and clean. Don't add comments unless they're necessary.
- Use proper abstraction only where truly required. Abstractions belong at the seams, not mid-code. Prefer plain, direct code so a change stays contained to one or two files.
- Design for reversibility: keep features self-contained, don't leak concerns across boundaries, and ask "what would it take to delete this?" before committing.

## Cross-platform
- All code targets Windows, Linux, and macOS: use `filepath`/`os` abstractions, put OS-specific code behind build tags, and never assume a POSIX shell, coreutils, or a case-sensitive filesystem.

## Concurrency
- Anything that blocks takes a `context.Context` and honors cancellation; every goroutine has a defined owner and exit path.
- Keep lock scopes minimal; never hold a lock across I/O, network, or LLM calls. Middleware wrapping `http.ResponseWriter` must preserve optional interfaces (`http.Flusher` etc.) — losing them silently kills SSE.

## Error handling
- Use the `golog` and `ctxerr` packages.
- Don't drop error checks. Define sentinel errors, wrap with context, and pass them up the call stack.
- Never `_ =` an error. Return + wrap when a caller can act on it; log at the call site only when a logger is already in scope; `panic` on invariants you don't expect to fail. Compile-time `var _ = X` and multi-return where the err slot is intentional are exempt.

## Data & persistence
- Evolve schemas through numbered, append-only migrations: `go:embed` a `migrations/` dir of sequential `NNNNNN_name.up.sql` and run them on startup in the store's `Open` (via the repo's existing runner — e.g. `internal/sqlmigrate` or `golang-migrate`). Scope data to its owner (one store/file per concern) so unrelated datasets can't collide.
- Relational: every table gets a primary key; store timestamps as Unix integers (`created_at`, `updated_at`); index what you filter/sort on; use foreign keys with cascading deletes; prefer an atomic upsert (`INSERT … ON CONFLICT … DO UPDATE`) over read-modify-write.
- Identifiers: for surrogate keys prefer a sortable ULID (or DB-native sequence) over a random UUIDv4 — random keys fragment index locality; use UUIDs only when an opaque, unguessable id is the point.
- Transactions: use them wherever it makes sense — any set of writes that must succeed or fail together belongs in one transaction. Keep them short, and don't hold one open across network/LLM calls.
- Vectors: store and key the index by the embedding model's identity (a dimension mismatch is corruption). A vector index is a derived cache — rebuildable from source, never the source of truth.
- Files: derive a stable on-disk key from a hash of the canonical path, not raw user input; for writes a crash could truncate, write atomically (temp file + rename, e.g. `fsutil.WriteFileAtomic`) rather than a bare `os.WriteFile`.
- Retention: keep only what the feature needs; make destructive actions explicit and confirmed; cap or prune anything that would grow unbounded. Migrations and deletes are one-way — design them safe to re-run.

## Datastar-specific gotchas
Follow only if Datastar is part of this repo's stack:
- Read signals (`datastar.ReadSignals`) before creating the SSE generator — `NewSSE` consumes the request body, so signals read after it come back empty.
- Don't rely on `data-show` for layout-critical visibility — toggle your own class; use `WithModeReplace` for full-element swaps.
- Datastar v1 has no execute-script SSE event, and `@post()` actions take no request body.

## Conventions
- Use the `Interface` suffix in interface names.
- Use table-driven tests as much as possible, with testify/require.
- Use `make lint`, `make test`, and `make build` (where present) — not ad-hoc go commands.
- Prefer the standard library; a new dependency must earn its place. Never bump version-paired dependencies without a smoke test.
- Before calling work done, run `make lint` and `make test` and exercise the changed behavior for real; report what you verified and what you couldn't.
- Where templ is part of the stack, generate with `go tool templ generate` (the version pinned in go.mod), never a globally-installed `templ` — a mismatched global binary silently rewrites the `_templ.go` version header and produces spurious diffs.
