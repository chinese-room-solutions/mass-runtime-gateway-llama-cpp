// Package huggingface provides HuggingFace-specific download helpers built on
// top of the generic download manager.
//
// The repo browser / search side of HuggingFace lives in mass-sdk
// (huggingface.SanitizeRepoID is canonical there). To keep this gateway
// self-contained we vendor a tiny SanitizeRepoID equivalent locally.
package huggingface

import (
	"context"
	"net/url"
	"path/filepath"
	"strings"

	"github.com/chinese-room-solutions/mass-runtime-llama-cpp/internal/download"
)

// hfTempSuffix is the temp file suffix used by HuggingFace downloads. Must
// match what TempFilePath produces so pause/cancel finds the same file.
const hfTempSuffix = ".downloading"

// SanitizeRepoID converts a HuggingFace repo ID to a directory-safe path.
// "publisher/repo" → "publisher/repo" (separators preserved as path segments).
// Mirrors mass-sdk/huggingface.SanitizeRepoID — kept tiny to avoid the
// cross-repo dep.
func SanitizeRepoID(repoID string) string {
	return strings.Trim(filepath.ToSlash(repoID), "/")
}

// Download fetches a file from HuggingFace into
// {modelsDir}/{sanitised_repo_id}/{filename}. Resumable on context cancel;
// the partial file is preserved and reused. Returns the absolute path to
// the downloaded file on success.
func Download(ctx context.Context, repoID, filename, modelsDir string, progressFn func(downloaded, total int64)) (string, error) {
	sanitised := SanitizeRepoID(repoID)
	destPath := filepath.Join(modelsDir, sanitised, filename)

	u := "https://huggingface.co/" + repoID + "/resolve/main/" + url.PathEscape(filename)

	opts := []download.Option{
		download.WithResume(true),
		download.WithMaxRetries(3),
		download.WithTempSuffix(hfTempSuffix),
	}
	if progressFn != nil {
		opts = append(opts, download.WithProgress(progressFn))
	}

	mgr := download.NewManager(nil)
	if err := mgr.Download(ctx, u, destPath, opts...); err != nil {
		return "", err
	}
	return destPath, nil
}

// TempFilePath returns the absolute path to the in-progress partial file.
// Useful for callers that want to remove it on cancel.
func TempFilePath(repoID, filename, modelsDir string) string {
	sanitised := SanitizeRepoID(repoID)
	destPath := filepath.Join(modelsDir, sanitised, filename)
	return download.TempFilePath(destPath, hfTempSuffix)
}

// StoreRelativePath returns the path the downloaded file will live at,
// relative to modelsDir. Suitable for use as a model_id input.
func StoreRelativePath(repoID, filename string) string {
	return filepath.ToSlash(filepath.Join(SanitizeRepoID(repoID), filename))
}
