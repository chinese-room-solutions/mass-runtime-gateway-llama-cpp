package gateway

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	gatewaypb "github.com/chinese-room-solutions/mass-proto/gen/go/gateway"
	"github.com/rs/zerolog"
	"github.com/stretchr/testify/require"
)

// storePathForID maps published ids (current-group-slug/<filename>) back
// to the physical under-format-root store path; everything else passes
// through unchanged.
func TestStorePathForID(t *testing.T) {
	cache := &metadataCache{entries: map[string]*catalogueEntry{
		// Group installed as "pdf2doc", later renamed to "Owen Embedding".
		"gguf/pdf2doc/primary-Q4_K_M.gguf": {Name: "Owen Embedding"},
		"gguf/pdf2doc/mmproj-f16.gguf":     {Name: "Owen Embedding", Companion: "mmproj"},
		// Group whose slug still matches its directory.
		"gguf/qwen-3/qwen3.gguf": {Name: "Qwen 3"},
	}}

	tests := []struct {
		name string
		id   string
		want string
	}{
		{"published id after rename", "owen-embedding/primary-Q4_K_M.gguf", "pdf2doc/primary-Q4_K_M.gguf"},
		{"companion resolves too", "owen-embedding/mmproj-f16.gguf", "pdf2doc/mmproj-f16.gguf"},
		{"slug matching its directory passes verbatim", "qwen-3/qwen3.gguf", "qwen-3/qwen3.gguf"},
		{"physical path of a renamed group passes through", "pdf2doc/primary-Q4_K_M.gguf", "pdf2doc/primary-Q4_K_M.gguf"},
		{"unknown slug passes through", "no-such-group/file.gguf", "no-such-group/file.gguf"},
		{"unknown filename in a known group passes through", "owen-embedding/other.gguf", "owen-embedding/other.gguf"},
		{"bare filename passes through", "file.gguf", "file.gguf"},
		{"deep path passes through", "a/b/file.gguf", "a/b/file.gguf"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			require.Equal(t, tt.want, cache.storePathForID(tt.id))
		})
	}
}

// groupRelPath keeps a group in one directory: new files join the
// directory the group's existing entries live in, even when the current
// name's slug differs (post-rename), so ids stay unambiguous and
// duplicate filenames surface as errDestTaken.
func TestGroupRelPathReusesGroupDir(t *testing.T) {
	cache := &metadataCache{entries: map[string]*catalogueEntry{
		"gguf/pdf2doc/primary.gguf": {Name: "Owen Embedding"},
	}}

	tests := []struct {
		name      string
		groupName string
		srcName   string
		want      string
	}{
		{"existing group keeps its directory", "Owen Embedding", "/src/extra.gguf", "gguf/pdf2doc/extra.gguf"},
		{"new group gets its slug directory", "Fresh Group", "/src/first.gguf", "gguf/fresh-group/first.gguf"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			require.Equal(t, tt.want, cache.groupRelPath(tt.groupName, tt.srcName))
		})
	}
}

// The published Model.id must re-derive from the group's CURRENT name:
// after a rename the same file lists under the new slug, while both the
// new and the old (physical) id keep resolving.
func TestPublishedIDTracksRename(t *testing.T) {
	modelsDir := t.TempDir()
	slugDir := filepath.Join(formatRoot(modelsDir), "pdf2doc")
	require.NoError(t, os.MkdirAll(slugDir, 0o755))
	abs := filepath.Join(slugDir, "primary-Q4_K_M.gguf")
	require.NoError(t, os.WriteFile(abs, make([]byte, 16), 0o644))
	st, err := os.Stat(abs)
	require.NoError(t, err)

	cache := newMetadataCache(t.TempDir(), modelsDir, zerolog.Nop())
	// A fully-warmed entry (capabilities, counts, hash, fresh mtime/size)
	// so parseModelInfo serves from the catalogue without reading a real
	// GGUF header.
	cache.entries["gguf/pdf2doc/primary-Q4_K_M.gguf"] = &catalogueEntry{
		Name:                 "pdf2doc",
		Capabilities:         &gatewaypb.Capabilities{},
		ParameterCount:       1,
		ActiveParameterCount: 1,
		Sha256:               "cafe",
		MTime:                st.ModTime(),
		Size:                 st.Size(),
	}

	info, ok := cache.parseModelInfo(abs, "pdf2doc/primary-Q4_K_M.gguf")
	require.True(t, ok)
	require.Equal(t, "pdf2doc/primary-Q4_K_M.gguf", info.ID)

	n, err := cache.renameGroup("pdf2doc", "Owen Embedding")
	require.NoError(t, err)
	require.Equal(t, 1, n)

	info, ok = cache.parseModelInfo(abs, "pdf2doc/primary-Q4_K_M.gguf")
	require.True(t, ok)
	require.Equal(t, "owen-embedding/primary-Q4_K_M.gguf", info.ID)

	// Both the published id and the physical path resolve to the file.
	g := &Gateway{modelsDir: modelsDir, cache: cache, logger: zerolog.Nop()}
	for _, id := range []string{"owen-embedding/primary-Q4_K_M.gguf", "pdf2doc/primary-Q4_K_M.gguf"} {
		resp, err := g.PlanDelete(context.Background(), &gatewaypb.PlanDeleteRequest{Id: id})
		require.NoError(t, err, id)
		require.ElementsMatch(t, []string{"gguf/pdf2doc/primary-Q4_K_M.gguf"}, resp.GetRelPaths(), id)
	}
}
