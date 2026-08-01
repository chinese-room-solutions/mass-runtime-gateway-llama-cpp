package gateway

import (
	"os"
	"path/filepath"
	"testing"

	llamacpp "github.com/chinese-room-solutions/mass-proto/gen/go/llama-cpp"
	workerpb "github.com/chinese-room-solutions/mass-proto/gen/go/worker"
	"github.com/stretchr/testify/require"
)

// buildLoadArtifacts must Stat the **absolute** on-disk path, not the
// store-relative ID — otherwise fileSize runs against the gateway's
// cwd and silently returns 0, which zeroes EstimatedLoadBytes for
// every job (the bug we hit in production).
func TestBuildLoadArtifacts_SizeBytesUsesAbsolutePath(t *testing.T) {
	modelsDir := t.TempDir()
	formatRootDir := filepath.Join(modelsDir, formatDir)
	require.NoError(t, os.MkdirAll(filepath.Join(formatRootDir, "qwen3"), 0o755))

	primary := filepath.Join(formatRootDir, "qwen3", "primary.gguf")
	mmproj := filepath.Join(formatRootDir, "qwen3", "mmproj.gguf")
	primaryBody := make([]byte, 4096)
	mmprojBody := make([]byte, 1024)
	require.NoError(t, os.WriteFile(primary, primaryBody, 0o644))
	require.NoError(t, os.WriteFile(mmproj, mmprojBody, 0o644))

	tests := []struct {
		name           string
		modelStr       string
		cfg            *loadConfig
		wantFiles      int
		wantPrimarySz  int64
		wantMmprojSz   int64
		wantPrimaryAbs string
		wantPrimaryKey string
		wantMmprojKey  string
	}{
		{
			name:           "primary only — SizeBytes resolved against absolute path",
			modelStr:       "qwen3/primary.gguf",
			cfg:            nil,
			wantFiles:      1,
			wantPrimarySz:  int64(len(primaryBody)),
			wantPrimaryAbs: primary,
			wantPrimaryKey: "gguf/qwen3/primary.gguf",
		},
		{
			name:           "primary + mmproj — both stat'd absolutely",
			modelStr:       "qwen3/primary.gguf",
			cfg:            &loadConfig{MmprojFilename: "mmproj.gguf"},
			wantFiles:      2,
			wantPrimarySz:  int64(len(primaryBody)),
			wantMmprojSz:   int64(len(mmprojBody)),
			wantPrimaryAbs: primary,
			wantPrimaryKey: "gguf/qwen3/primary.gguf",
			wantMmprojKey:  "gguf/qwen3/mmproj.gguf",
		},
	}

	h := &handlers{modelsDir: modelsDir}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, files, err := h.buildLoadArtifacts(tt.modelStr, tt.cfg, llamacpp.LoadKind_LOAD_KIND_CHAT)
			require.NoError(t, err)
			require.Len(t, files, tt.wantFiles)

			var primaryFile, mmprojFile *workerpb.ModelFile
			for _, f := range files {
				switch f.GetRole() {
				case workerpb.ModelFileRole_MODEL_FILE_ROLE_PRIMARY:
					primaryFile = f
				case workerpb.ModelFileRole_MODEL_FILE_ROLE_MMPROJ:
					mmprojFile = f
				}
			}
			require.NotNil(t, primaryFile)
			require.Equal(t, tt.wantPrimaryAbs, primaryFile.GetLocalPath())
			require.Equal(t, tt.wantPrimaryKey, primaryFile.GetFilename(),
				"Filename must be the full store-relative key incl. the runtime-owned first segment")
			require.Equal(t, tt.wantPrimarySz, primaryFile.GetSizeBytes(),
				"primary SizeBytes must be the real file size, not 0 from a relative-path stat")
			if tt.wantFiles == 2 {
				require.NotNil(t, mmprojFile)
				require.Equal(t, tt.wantMmprojKey, mmprojFile.GetFilename())
				require.Equal(t, tt.wantMmprojSz, mmprojFile.GetSizeBytes())
			}
		})
	}
}

// A CHAT load auto-attaches the sibling mmproj projector the gateway
// knows about (same operator-typed Name) even when the client sends no
// MmprojFilename — clients must not have to know companion filenames.
// This is the fix for vision apps (pdf2doc) silently loading text-only
// and erroring with "multimodal request but model loaded without mmproj".
func TestBuildLoadArtifacts_AutoAttachesSiblingMmproj(t *testing.T) {
	modelsDir := t.TempDir()
	formatRootDir := filepath.Join(modelsDir, formatDir)
	require.NoError(t, os.MkdirAll(filepath.Join(formatRootDir, "pdf2doc"), 0o755))

	primary := filepath.Join(formatRootDir, "pdf2doc", "pdf2doc-Q4_K_M.gguf")
	mmproj := filepath.Join(formatRootDir, "pdf2doc", "mmproj-pdf2doc-f16.gguf")
	require.NoError(t, os.WriteFile(primary, make([]byte, 4096), 0o644))
	require.NoError(t, os.WriteFile(mmproj, make([]byte, 1024), 0o644))

	// Catalogue: both files share the operator Name; the projector row
	// is tagged as the mmproj companion. Keys are modelsDir-relative.
	cache := &metadataCache{modelsDir: modelsDir, entries: map[string]*catalogueEntry{
		"gguf/pdf2doc/pdf2doc-Q4_K_M.gguf":     {Name: "pdf2doc"},
		"gguf/pdf2doc/mmproj-pdf2doc-f16.gguf": {Name: "pdf2doc", Companion: "mmproj"},
	}}
	h := &handlers{modelsDir: modelsDir, cache: cache}

	t.Run("chat load auto-attaches mmproj", func(t *testing.T) {
		_, files, err := h.buildLoadArtifacts(
			"pdf2doc/pdf2doc-Q4_K_M.gguf", nil, llamacpp.LoadKind_LOAD_KIND_CHAT)
		require.NoError(t, err)
		require.Len(t, files, 2)
		var hasMmproj bool
		for _, f := range files {
			if f.GetRole() == workerpb.ModelFileRole_MODEL_FILE_ROLE_MMPROJ {
				hasMmproj = true
				require.Equal(t, mmproj, f.GetLocalPath())
				require.Equal(t, "gguf/pdf2doc/mmproj-pdf2doc-f16.gguf", f.GetFilename())
			}
		}
		require.True(t, hasMmproj, "vision chat must auto-attach the projector")
	})

	t.Run("embedding load does not attach mmproj", func(t *testing.T) {
		_, files, err := h.buildLoadArtifacts(
			"pdf2doc/pdf2doc-Q4_K_M.gguf", nil, llamacpp.LoadKind_LOAD_KIND_EMBEDDING)
		require.NoError(t, err)
		require.Len(t, files, 1)
	})
}

// buildLoadArtifacts stamps ModelFile.sha256 from the catalogue so a
// remote worker can verify integrity. A missing entry, a nil cache, or
// an unhashable file leaves the field empty — the worker then skips
// verification (safe degrade).
func TestBuildLoadArtifacts_Sha256FromCatalogue(t *testing.T) {
	modelsDir := t.TempDir()
	formatRootDir := filepath.Join(modelsDir, formatDir)
	require.NoError(t, os.MkdirAll(filepath.Join(formatRootDir, "qwen3"), 0o755))

	primary := filepath.Join(formatRootDir, "qwen3", "primary.gguf")
	mmproj := filepath.Join(formatRootDir, "qwen3", "mmproj.gguf")
	require.NoError(t, os.WriteFile(primary, make([]byte, 4096), 0o644))
	require.NoError(t, os.WriteFile(mmproj, make([]byte, 1024), 0o644))

	const primarySha = "1111111111111111111111111111111111111111111111111111111111111111"
	const mmprojSha = "2222222222222222222222222222222222222222222222222222222222222222"

	// Cache pre-populated with digests — the fast path returns them
	// without re-hashing the files.
	cacheWithSha := &metadataCache{modelsDir: modelsDir, entries: map[string]*catalogueEntry{
		"gguf/qwen3/primary.gguf": {Name: "qwen3", Sha256: primarySha},
		"gguf/qwen3/mmproj.gguf":  {Name: "qwen3", Companion: "mmproj", Sha256: mmprojSha},
	}}

	tests := []struct {
		name           string
		h              *handlers
		cfg            *loadConfig
		wantPrimarySha string
		wantMmprojSha  string
	}{
		{
			name:           "digests populated from catalogue",
			h:              &handlers{modelsDir: modelsDir, cache: cacheWithSha},
			cfg:            &loadConfig{MmprojFilename: "mmproj.gguf"},
			wantPrimarySha: primarySha,
			wantMmprojSha:  mmprojSha,
		},
		{
			// No cache wired: field stays empty rather than NPE.
			name:           "nil cache leaves sha256 empty",
			h:              &handlers{modelsDir: modelsDir},
			cfg:            &loadConfig{MmprojFilename: "mmproj.gguf"},
			wantPrimarySha: "",
			wantMmprojSha:  "",
		},
		{
			// Cache present but no entry for this file → empty digest.
			name:           "unknown entry leaves sha256 empty",
			h:              &handlers{modelsDir: modelsDir, cache: &metadataCache{modelsDir: modelsDir, entries: map[string]*catalogueEntry{}}},
			cfg:            &loadConfig{MmprojFilename: "mmproj.gguf"},
			wantPrimarySha: "",
			wantMmprojSha:  "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, files, err := tt.h.buildLoadArtifacts("qwen3/primary.gguf", tt.cfg, llamacpp.LoadKind_LOAD_KIND_CHAT)
			require.NoError(t, err)
			require.Len(t, files, 2)

			var primaryFile, mmprojFile *workerpb.ModelFile
			for _, f := range files {
				switch f.GetRole() {
				case workerpb.ModelFileRole_MODEL_FILE_ROLE_PRIMARY:
					primaryFile = f
				case workerpb.ModelFileRole_MODEL_FILE_ROLE_MMPROJ:
					mmprojFile = f
				}
			}
			require.NotNil(t, primaryFile)
			require.NotNil(t, mmprojFile)
			require.Equal(t, tt.wantPrimarySha, primaryFile.GetSha256())
			require.Equal(t, tt.wantMmprojSha, mmprojFile.GetSha256())
		})
	}
}

// fileSize must return the real size for an absolute path and 0 (no
// panic) for a non-existent one. Documents the contract MASS relies
// on: a stat failure is a graceful "unknown," not an error.
func TestFileSize(t *testing.T) {
	tmp := t.TempDir()
	real := filepath.Join(tmp, "real.bin")
	require.NoError(t, os.WriteFile(real, []byte("hello"), 0o644))

	tests := []struct {
		name string
		path string
		want int64
	}{
		{"real file returns its size", real, 5},
		{"missing file returns 0", filepath.Join(tmp, "missing.bin"), 0},
		{"empty path returns 0", "", 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			require.Equal(t, tt.want, fileSize(tt.path))
		})
	}
}
