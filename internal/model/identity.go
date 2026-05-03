// Package model derives stable model_ids and resolves request "model"
// strings to on-disk model store paths.
//
// model_id is gateway-defined and opaque to MASS; we use:
//
//	"<store-relative-path>#<config-hash>"
//
// where config-hash is a short SHA-256 of the load-time config (context
// size, gpu_layers override, mmproj path, ...). Two requests asking for the
// same file with the same config get the same model_id and hit the same
// loaded worker instance; differing configs spawn fresh instances.
package model

import (
	"crypto/sha256"
	"encoding/hex"
	"path"
	"strings"

	llamacpp "github.com/chinese-room-solutions/mass-proto/gen/go/llama-cpp"
	"google.golang.org/protobuf/proto"
)

// ID derives the canonical model_id from a relative store path + load hints.
// hints may be nil (then config-hash is empty and the id is just the path).
func ID(storePath string, hints *llamacpp.LoadHints) string {
	storePath = strings.TrimPrefix(strings.ReplaceAll(storePath, "\\", "/"), "/")
	if hints == nil {
		return storePath
	}
	b, err := proto.Marshal(hints)
	if err != nil || len(b) == 0 {
		return storePath
	}
	sum := sha256.Sum256(b)
	return storePath + "#" + hex.EncodeToString(sum[:6])
}

// ResolveModelPath turns the user-facing "model" string from a request into
// a store-relative path suitable for ID and EnsureModelLoaded.
//
// Accepts:
//   - "publisher/repo/file.gguf"     (HF-style id; treated as a relative path)
//   - "subdir/file.gguf"             (any store-relative path)
//   - "file.gguf"                    (top-level file)
//
// Rejected (returns ""):
//   - absolute paths
//   - paths containing ".."
func ResolveModelPath(model string) string {
	if model == "" {
		return ""
	}
	clean := path.Clean(strings.ReplaceAll(model, "\\", "/"))
	if clean == "." || strings.HasPrefix(clean, "/") || strings.HasPrefix(clean, "../") || clean == ".." {
		return ""
	}
	return clean
}
