package gateway

import (
	"os"
	"path/filepath"
	"testing"

	gatewaypb "github.com/chinese-room-solutions/mass-proto/gen/go/gateway"
	"github.com/rs/zerolog"
	"github.com/stretchr/testify/require"
)

// modelTypeFromHeader prefers pooling_type over chat_template
// because some embedding models (Qwen3-Embedding) ship both.
// Projectors return "" (not a standalone model_type).
func TestModelTypeFromHeader(t *testing.T) {
	tests := []struct {
		name string
		kv   map[string]string
		want string
	}{
		{name: "clip projector returns empty", kv: map[string]string{"architecture": "clip"}, want: ""},
		{name: "chat template present, no pooling means chat", kv: map[string]string{"architecture": "qwen3", "chat_template_present": "true"}, want: "chat"},
		{name: "no signals defaults to embedding", kv: map[string]string{"architecture": "bert"}, want: "embedding"},
		{name: "pooling_type beats chat_template (Qwen3-Embedding case)", kv: map[string]string{"architecture": "qwen3", "chat_template_present": "true", "pooling_type_present": "true"}, want: "embedding"},
		{name: "pooling_type alone means embedding", kv: map[string]string{"architecture": "bert", "pooling_type_present": "true"}, want: "embedding"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			require.Equal(t, tt.want, modelTypeFromHeader(tt.kv))
		})
	}
}

// Thinking is gated on the file being a chat model — embedding
// models (Qwen3-Embedding) ship a chat template containing think
// tokens but don't actually emit reasoning, so the flag would lie.
func TestCapabilitiesFromHeader_Thinking(t *testing.T) {
	tests := []struct {
		name string
		kv   map[string]string
		want bool
	}{
		{name: "chat model with thinking template", kv: map[string]string{"thinking": "true", "chat_template_present": "true"}, want: true},
		{name: "embedding model with thinking template (Qwen3-Embedding)", kv: map[string]string{"thinking": "true", "chat_template_present": "true", "pooling_type_present": "true"}, want: false},
		{name: "chat model without thinking", kv: map[string]string{"chat_template_present": "true"}, want: false},
		{name: "no signals", kv: map[string]string{}, want: false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			require.Equal(t, tt.want, capabilitiesFromHeader(tt.kv).GetThinking())
		})
	}
}

func TestLooksLikeMmprojFilename(t *testing.T) {
	tests := []struct {
		name string
		want bool
	}{
		{name: "mmproj-Qwen.gguf", want: true},
		{name: "Qwen-mmproj-F16.gguf", want: true},
		{name: "qwen-7b-Q4.gguf", want: false},
		{name: "mmproj.txt", want: false},
		{name: "model.gguf", want: false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			require.Equal(t, tt.want, looksLikeMmprojFilename(tt.name))
		})
	}
}

// modelSlug round-trips through nameForSlug: same Name produces the
// same slug, distinct Names produce distinct slugs.
func TestModelSlug(t *testing.T) {
	tests := []struct {
		name string
		want string
	}{
		{name: "GLM-4.6V-Flash", want: "glm-4-6v-flash"},
		{name: "Qwen3 7B", want: "qwen3-7b"},
		{name: "  spaces  ", want: "spaces"},
		{name: "many   spaces", want: "many-spaces"},
		{name: "Trailing-Punct!!!", want: "trailing-punct"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			require.Equal(t, tt.want, modelSlug(tt.name))
		})
	}
	require.Equal(t, modelSlug("MyModel"), modelSlug("MyModel"))
	require.NotEqual(t, modelSlug("A"), modelSlug("B"))
}

// groupModels groups catalogue entries by exact-match Name.
// Same Name → same Group. Each catalogue entry becomes a child Model.
func TestGroupModels_NameGrouping(t *testing.T) {
	infos := []*parsedModel{
		{ID: "qwen-q4.gguf", Name: "Qwen3 7B", VariantLabel: "Q4_K_M", ModelType: "chat", SizeBytes: 1, Capabilities: &gatewaypb.Capabilities{}},
		{ID: "qwen-q8.gguf", Name: "Qwen3 7B", VariantLabel: "Q8_0", ModelType: "chat", SizeBytes: 2, Capabilities: &gatewaypb.Capabilities{}},
		{ID: "llama.gguf", Name: "Llama 3.1", VariantLabel: "Q4_K_M", ModelType: "chat", SizeBytes: 3, Capabilities: &gatewaypb.Capabilities{}},
	}
	out := groupModels(infos)
	require.Len(t, out, 2)
	// Sorted by display name.
	require.Equal(t, "Llama 3.1", out[0].GetDisplayName())
	require.Equal(t, "Qwen3 7B", out[1].GetDisplayName())
	// Qwen has two child models.
	require.Len(t, out[1].GetModels(), 2)
}

func TestGroupModels_ProjectorIsItsOwnModel(t *testing.T) {
	infos := []*parsedModel{
		{ID: "glm-q4.gguf", Name: "GLM-4.6V", VariantLabel: "Q4_K_M", ModelType: "chat", SizeBytes: 1, Capabilities: &gatewaypb.Capabilities{Vision: true}},
		{ID: "glm-mmproj.gguf", Name: "GLM-4.6V", Companion: "mmproj", VariantLabel: "Mmproj", SizeBytes: 2, Capabilities: &gatewaypb.Capabilities{Vision: true}},
	}
	out := groupModels(infos)
	require.Len(t, out, 1, "chat + projector under same name → one group")
	require.Len(t, out[0].GetModels(), 2, "every file is its own model row")
	require.True(t, out[0].GetCapabilities().GetVision(), "projector contributes vision capability to the group union")
	require.Equal(t, []string{"chat"}, out[0].GetModelTypes(), "non-companion files contribute their model_type")
}

// Distinct non-companion model_types in one group surface as a list
// in insertion order (so MASS can render one badge per type).
// Companions (projectors) don't contribute a type.
func TestGroupModels_CollectsDistinctModelTypes(t *testing.T) {
	infos := []*parsedModel{
		{ID: "a.gguf", Name: "Mixed", ModelType: "chat", Capabilities: &gatewaypb.Capabilities{}},
		{ID: "b.gguf", Name: "Mixed", ModelType: "chat", Capabilities: &gatewaypb.Capabilities{}}, // duplicate type
		{ID: "c.gguf", Name: "Mixed", ModelType: "embedding", Capabilities: &gatewaypb.Capabilities{}},
		{ID: "d.gguf", Name: "Mixed", Companion: "mmproj", Capabilities: &gatewaypb.Capabilities{}}, // ignored
	}
	out := groupModels(infos)
	require.Len(t, out, 1)
	require.Equal(t, []string{"chat", "embedding"}, out[0].GetModelTypes())
}

func TestGroupModels_EmitsSlugAsID(t *testing.T) {
	infos := []*parsedModel{
		{ID: "x.gguf", Name: "My Model", ModelType: "chat", Capabilities: &gatewaypb.Capabilities{}},
	}
	out := groupModels(infos)
	require.Len(t, out, 1)
	require.Equal(t, modelSlug("My Model"), out[0].GetId())
}

// renameGroup rewrites every entry whose Name matches old AND moves
// the underlying files into the new slug's subdirectory under
// gguf/. Files are required to be committed (header read); entries
// without Capabilities (download in flight) refuse the rename.
func TestRenameGroup(t *testing.T) {
	tmp := t.TempDir()
	formatRoot := filepath.Join(tmp, "gguf")
	require.NoError(t, os.MkdirAll(filepath.Join(formatRoot, modelSlug("Old Name")), 0o755))
	require.NoError(t, os.MkdirAll(filepath.Join(formatRoot, modelSlug("Other")), 0o755))

	pathA := filepath.Join(formatRoot, modelSlug("Old Name"), "a.gguf")
	pathB := filepath.Join(formatRoot, modelSlug("Old Name"), "b.gguf")
	pathC := filepath.Join(formatRoot, modelSlug("Other"), "c.gguf")
	for _, p := range []string{pathA, pathB, pathC} {
		require.NoError(t, os.WriteFile(p, []byte("dummy"), 0o644))
	}

	c := &metadataCache{
		logger:  zerolog.Nop(),
		entries: map[string]*catalogueEntry{},
	}
	c.entries[pathA] = &catalogueEntry{Name: "Old Name", Capabilities: &gatewaypb.Capabilities{}}
	c.entries[pathB] = &catalogueEntry{Name: "Old Name", Capabilities: &gatewaypb.Capabilities{}}
	c.entries[pathC] = &catalogueEntry{Name: "Other", Capabilities: &gatewaypb.Capabilities{}}

	n, err := c.renameGroup(tmp, "Old Name", "New Name")
	require.NoError(t, err)
	require.Equal(t, 2, n)

	newA := filepath.Join(formatRoot, modelSlug("New Name"), "a.gguf")
	newB := filepath.Join(formatRoot, modelSlug("New Name"), "b.gguf")
	require.FileExists(t, newA)
	require.FileExists(t, newB)
	require.NoFileExists(t, pathA)
	require.NoFileExists(t, pathB)
	require.Equal(t, "New Name", c.entries[newA].Name)
	require.Equal(t, "New Name", c.entries[newB].Name)
	require.NotContains(t, c.entries, pathA, "old map key removed")
	require.Equal(t, "Other", c.entries[pathC].Name, "untouched")

	// No-ops.
	for _, args := range [][2]string{
		{"missing", "anything"},
		{"Other", "Other"},
		{"", "X"},
		{"Other", ""},
	} {
		got, err := c.renameGroup(tmp, args[0], args[1])
		require.NoError(t, err)
		require.Zero(t, got)
	}
}

// renameGroup refuses when any matching entry is reserved (no header
// data yet — download in flight).
func TestRenameGroup_RefusesReserved(t *testing.T) {
	tmp := t.TempDir()
	c := &metadataCache{
		logger:  zerolog.Nop(),
		entries: map[string]*catalogueEntry{},
	}
	c.entries["/m/in-flight.gguf"] = &catalogueEntry{Name: "Pending"} // Capabilities=nil

	n, err := c.renameGroup(tmp, "Pending", "Done")
	require.ErrorIs(t, err, errEntryReserved)
	require.Zero(t, n)
}

// nameForSlug looks up the source Name from its slug.
func TestNameForSlug(t *testing.T) {
	c := &metadataCache{
		logger:  zerolog.Nop(),
		entries: map[string]*catalogueEntry{},
	}
	c.entries["/m/a.gguf"] = &catalogueEntry{Name: "Qwen3 7B"}
	c.entries["/m/b.gguf"] = &catalogueEntry{Name: "Llama 3.1"}

	require.Equal(t, "Qwen3 7B", c.nameForSlug("qwen3-7b"))
	require.Equal(t, "Llama 3.1", c.nameForSlug("llama-3-1"))
	require.Equal(t, "", c.nameForSlug("missing"))
}

// pruneAbsent must keep reserved entries (no Capabilities yet) so a
// PlanLocalImport reservation written before the file lands on disk
// survives the next walk. Only committed entries (header successfully
// read) get pruned when their file is missing.
func TestPruneAbsent_KeepsReservedEntries(t *testing.T) {
	c := &metadataCache{
		logger:  zerolog.Nop(),
		entries: map[string]*catalogueEntry{},
	}
	// Reserved: just-typed name, no header yet (file not on disk).
	c.entries["/m/in-flight.gguf"] = &catalogueEntry{Name: "Pending"}
	// Committed: header read, file vanished out-of-band.
	c.entries["/m/gone.gguf"] = &catalogueEntry{
		Name:         "Was-There",
		Capabilities: &gatewaypb.Capabilities{},
	}
	// Committed: still on disk.
	c.entries["/m/here.gguf"] = &catalogueEntry{
		Name:         "Here",
		Capabilities: &gatewaypb.Capabilities{},
	}

	c.pruneAbsent(map[string]struct{}{"/m/here.gguf": {}})

	_, kept := c.entries["/m/in-flight.gguf"]
	require.True(t, kept, "reserved entry must survive prune")
	_, gone := c.entries["/m/gone.gguf"]
	require.False(t, gone, "committed entry whose file is missing should be pruned")
	_, here := c.entries["/m/here.gguf"]
	require.True(t, here, "committed entry whose file is observed should stay")
}

// distinctNames powers the install-dialog autocomplete.
func TestDistinctNames(t *testing.T) {
	c := &metadataCache{
		logger:  zerolog.Nop(),
		entries: map[string]*catalogueEntry{},
	}
	c.entries["/m/a.gguf"] = &catalogueEntry{Name: "Qwen"}
	c.entries["/m/b.gguf"] = &catalogueEntry{Name: "Qwen"} // duplicate
	c.entries["/m/c.gguf"] = &catalogueEntry{Name: "Llama"}
	c.entries["/m/d.gguf"] = &catalogueEntry{Name: ""} // skipped

	got := c.distinctNames()
	require.ElementsMatch(t, []string{"Qwen", "Llama"}, got)
}
