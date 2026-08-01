package main

import (
	"archive/zip"
	"io"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
	"gopkg.in/yaml.v3"
)

// The packaged runtime.yml must point at the binary actually archived:
// on Windows the build produces bin/mass-runtime-gateway-llama-cpp.exe while
// the static manifest says bin/mass-runtime-gateway-llama-cpp — shipping the
// manifest verbatim makes a broken package MASS can't launch.
func TestRewriteBinaryField(t *testing.T) {
	tests := []struct {
		name     string
		manifest string
		entry    string
	}{
		{
			name: "posix name rewritten to exe",
			manifest: "runtime_name: llama-cpp\n" +
				"binary: bin/mass-runtime-gateway-llama-cpp\n",
			entry: "bin/mass-runtime-gateway-llama-cpp.exe",
		},
		{
			name: "same name is a no-op rewrite",
			manifest: "runtime_name: llama-cpp\n" +
				"binary: bin/mass-runtime-gateway-llama-cpp\n",
			entry: "bin/mass-runtime-gateway-llama-cpp",
		},
		{
			name:     "missing binary field is added",
			manifest: "runtime_name: llama-cpp\n",
			entry:    "bin/mass-runtime-gateway-llama-cpp.exe",
		},
		{
			name: "other fields survive untouched",
			manifest: "runtime_name: llama-cpp\n" +
				"version: 0.1.0\n" +
				"description: |\n" +
				"  Multi-line\n" +
				"  description.\n" +
				"binary: bin/old-name\n",
			entry: "bin/mass-runtime-gateway-llama-cpp.exe",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			out, err := rewriteBinaryField([]byte(tt.manifest), tt.entry)
			require.NoError(t, err)

			var got map[string]any
			require.NoError(t, yaml.Unmarshal(out, &got))
			require.Equal(t, tt.entry, got["binary"])
			require.Equal(t, "llama-cpp", got["runtime_name"], "untouched fields must survive")

			var want map[string]any
			require.NoError(t, yaml.Unmarshal([]byte(tt.manifest), &want))
			want["binary"] = tt.entry
			require.Equal(t, want, got, "rewrite must only change the binary field")
		})
	}
}

func TestRewriteBinaryField_RejectsNonMapping(t *testing.T) {
	_, err := rewriteBinaryField([]byte("- just\n- a\n- list\n"), "bin/x")
	require.Error(t, err)
}

// buildArchive end-to-end: the archive holds runtime.yml (with the
// binary field pointing at the packed entry) and the binary itself
// under bin/<basename>.
func TestBuildArchive(t *testing.T) {
	tmp := t.TempDir()
	binaryPath := filepath.Join(tmp, "mass-runtime-gateway-llama-cpp.exe")
	require.NoError(t, os.WriteFile(binaryPath, []byte("fake-binary"), 0o755))
	manifestPath := filepath.Join(tmp, "runtime.yml")
	require.NoError(t, os.WriteFile(manifestPath, []byte(
		"runtime_name: llama-cpp\nbinary: bin/mass-runtime-gateway-llama-cpp\n"), 0o644))
	outPath := filepath.Join(tmp, "out.mass")

	require.NoError(t, buildArchive(outPath, binaryPath, manifestPath))

	r, err := zip.OpenReader(outPath)
	require.NoError(t, err)
	defer func() { _ = r.Close() }()

	entries := map[string]*zip.File{}
	for _, f := range r.File {
		entries[f.Name] = f
	}
	require.Contains(t, entries, "runtime.yml")
	require.Contains(t, entries, "bin/mass-runtime-gateway-llama-cpp.exe")

	rc, err := entries["runtime.yml"].Open()
	require.NoError(t, err)
	defer func() { _ = rc.Close() }()
	buf, err := io.ReadAll(rc)
	require.NoError(t, err)

	var manifest map[string]any
	require.NoError(t, yaml.Unmarshal(buf, &manifest))
	require.Equal(t, "bin/mass-runtime-gateway-llama-cpp.exe", manifest["binary"],
		"packaged manifest must point at the archived binary name")

	require.Equal(t, os.FileMode(0o755), entries["bin/mass-runtime-gateway-llama-cpp.exe"].Mode().Perm())
}
