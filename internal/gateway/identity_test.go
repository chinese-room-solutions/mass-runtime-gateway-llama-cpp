package gateway

import (
	"os"
	"path/filepath"
	"testing"
	"time"

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
	require.Equal(t, []gatewaypb.ModelTypeKind{gatewaypb.ModelTypeKind_MODEL_TYPE_CHAT}, modelTypeKinds(out[0]),
		"non-companion files contribute their model_type")
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
	require.Equal(t, []gatewaypb.ModelTypeKind{
		gatewaypb.ModelTypeKind_MODEL_TYPE_CHAT,
		gatewaypb.ModelTypeKind_MODEL_TYPE_EMBEDDING,
	}, modelTypeKinds(out[0]))
}

// modelTypeKinds extracts the typed kinds from a Group's model_types
// for terse equality checks in tests.
func modelTypeKinds(g *gatewaypb.Group) []gatewaypb.ModelTypeKind {
	entries := g.GetModelTypes()
	out := make([]gatewaypb.ModelTypeKind, len(entries))
	for i, e := range entries {
		out[i] = e.GetKind()
	}
	return out
}

func TestGroupModels_EmitsSlugAsID(t *testing.T) {
	infos := []*parsedModel{
		{ID: "x.gguf", Name: "My Model", ModelType: "chat", Capabilities: &gatewaypb.Capabilities{}},
	}
	out := groupModels(infos)
	require.Len(t, out, 1)
	require.Equal(t, modelSlug("My Model"), out[0].GetId())
}

// renameGroup rewrites every entry whose Name matches old — and ONLY
// the catalogue: files never move, because the store-relative path is
// the model_id MASS and loaded workers key on. Keys stay stable, disk
// stays untouched.
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
		logger:    zerolog.Nop(),
		modelsDir: tmp,
		entries:   map[string]*catalogueEntry{},
	}
	keyFor := func(abs string) string {
		key, ok := c.entryKey(abs)
		require.True(t, ok)
		return key
	}
	c.entries[keyFor(pathA)] = &catalogueEntry{Name: "Old Name", Capabilities: &gatewaypb.Capabilities{}}
	c.entries[keyFor(pathB)] = &catalogueEntry{Name: "Old Name", Capabilities: &gatewaypb.Capabilities{}}
	c.entries[keyFor(pathC)] = &catalogueEntry{Name: "Other", Capabilities: &gatewaypb.Capabilities{}}

	n, err := c.renameGroup("Old Name", "New Name")
	require.NoError(t, err)
	require.Equal(t, 2, n)

	// Files stay in place, catalogue keys stay stable.
	require.FileExists(t, pathA)
	require.FileExists(t, pathB)
	require.NoDirExists(t, filepath.Join(formatRoot, modelSlug("New Name")), "rename must not create a new slug dir")
	require.Equal(t, "New Name", c.entries[keyFor(pathA)].Name)
	require.Equal(t, "New Name", c.entries[keyFor(pathB)].Name)
	require.Equal(t, "Other", c.entries[keyFor(pathC)].Name, "untouched")
	require.True(t, c.dirty, "rename must mark the catalogue for persistence")

	// No-ops.
	for _, args := range [][2]string{
		{"missing", "anything"},
		{"Other", "Other"},
		{"", "X"},
		{"Other", ""},
	} {
		got, err := c.renameGroup(args[0], args[1])
		require.NoError(t, err)
		require.Zero(t, got)
	}
}

// Reserved entries (download in flight, no header yet) rename freely:
// only the catalogue Name changes, no bytes move, so there is nothing
// to orphan. The in-flight download still lands at its reserved path
// and joins the group under its new name.
func TestRenameGroup_RenamesReservedEntries(t *testing.T) {
	tmp := t.TempDir()
	c := &metadataCache{
		logger:    zerolog.Nop(),
		modelsDir: tmp,
		entries:   map[string]*catalogueEntry{},
	}
	c.entries["gguf/pending/in-flight.gguf"] = &catalogueEntry{Name: "Pending"} // Capabilities=nil

	n, err := c.renameGroup("Pending", "Done")
	require.NoError(t, err)
	require.Equal(t, 1, n)
	require.Equal(t, "Done", c.entries["gguf/pending/in-flight.gguf"].Name)
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
// survives the next walk. Committed entries are only pruned when the
// file is confirmed gone by os.Stat — an entry whose file exists but
// failed to parse this walk (momentary AV lock, transient read error)
// must survive, or the operator-typed Name would be erased forever.
func TestPruneAbsent_StatGatedPrune(t *testing.T) {
	modelsDir := t.TempDir()
	c := &metadataCache{
		logger:    zerolog.Nop(),
		modelsDir: modelsDir,
		entries:   map[string]*catalogueEntry{},
	}
	// Unreadable-but-present: file on disk, header read failed, so it's
	// not in observed.
	require.NoError(t, os.MkdirAll(filepath.Join(modelsDir, "m"), 0o755))
	require.NoError(t, os.WriteFile(filepath.Join(modelsDir, "m", "locked.gguf"), []byte("x"), 0o644))
	c.entries["m/locked.gguf"] = &catalogueEntry{
		Name:         "Locked",
		Capabilities: &gatewaypb.Capabilities{},
	}
	// Reserved: just-typed name, no header yet (file not on disk).
	c.entries["m/in-flight.gguf"] = &catalogueEntry{Name: "Pending"}
	// Committed: header read, file vanished out-of-band.
	c.entries["m/gone.gguf"] = &catalogueEntry{
		Name:         "Was-There",
		Capabilities: &gatewaypb.Capabilities{},
	}
	// Committed: still on disk and observed.
	c.entries["m/here.gguf"] = &catalogueEntry{
		Name:         "Here",
		Capabilities: &gatewaypb.Capabilities{},
	}

	c.pruneAbsent(map[string]struct{}{"m/here.gguf": {}})

	_, kept := c.entries["m/in-flight.gguf"]
	require.True(t, kept, "reserved entry must survive prune")
	_, gone := c.entries["m/gone.gguf"]
	require.False(t, gone, "committed entry whose file is missing should be pruned")
	_, here := c.entries["m/here.gguf"]
	require.True(t, here, "committed entry whose file is observed should stay")
	_, locked := c.entries["m/locked.gguf"]
	require.True(t, locked, "entry whose file exists but failed to parse must be kept for retry")
}

// Group.id = modelSlug(Name), so two DIFFERENT names sharing a slug
// make delete/rename resolve arbitrarily — reserveEntry must reject
// the collision (and names that slug to nothing) at plan time.
func TestReserveEntry_SlugUniqueness(t *testing.T) {
	tests := []struct {
		name     string
		existing string // Name of a pre-existing entry ("" = none)
		reserve  string
		wantErr  error
	}{
		{name: "distinct slug accepted", existing: "Qwen 3", reserve: "Llama 3", wantErr: nil},
		{name: "same name accepted (files join the group)", existing: "Qwen 3", reserve: "Qwen 3", wantErr: nil},
		{name: "different name, same slug rejected", existing: "Qwen 3", reserve: "qwen-3", wantErr: errSlugCollision},
		{name: "empty slug rejected", reserve: "北京", wantErr: errNameNotSluggable},
		{name: "punctuation-only name rejected", reserve: "!!!", wantErr: errNameNotSluggable},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			modelsDir := t.TempDir()
			c := &metadataCache{
				logger:    zerolog.Nop(),
				modelsDir: modelsDir,
				path:      filepath.Join(t.TempDir(), catalogueFilename),
				entries:   map[string]*catalogueEntry{},
			}
			if tt.existing != "" {
				c.entries["gguf/existing/a.gguf"] = &catalogueEntry{Name: tt.existing}
			}
			err := c.reserveEntry(filepath.Join(modelsDir, "gguf", "new", "b.gguf"), tt.reserve)
			if tt.wantErr == nil {
				require.NoError(t, err)
				require.Contains(t, c.entries, "gguf/new/b.gguf")
			} else {
				require.ErrorIs(t, err, tt.wantErr)
				require.NotContains(t, c.entries, "gguf/new/b.gguf", "failed reserve must not leave an entry behind")
			}
		})
	}
}

// reserveEntry must fail when the destination is already occupied — a
// file on disk, or a live catalogue entry claimed by a DIFFERENT group
// (e.g. a reservation for a download still in flight). This closes the
// plan-time TOCTOU: the check runs under the cache lock, so two
// concurrent plans for the same destination can't both win. Re-plans of
// the same (dest, name) pair stay allowed.
func TestReserveEntry_DestConflicts(t *testing.T) {
	tests := []struct {
		name       string
		existing   *catalogueEntry // pre-seeded at the destination key ("" = none)
		fileOnDisk bool            // destination file exists
		reserve    string          // group name being reserved
		wantErr    error
	}{
		{
			name:    "fresh destination accepted",
			reserve: "Qwen 3",
		},
		{
			name:     "same-name re-plan re-stamps the reservation",
			existing: &catalogueEntry{Name: "Qwen 3"},
			reserve:  "Qwen 3",
		},
		{
			name:     "live reservation by another group rejected",
			existing: &catalogueEntry{Name: "Other Group"},
			reserve:  "Qwen 3",
			wantErr:  errDestTaken,
		},
		{
			name:       "file already on disk rejected",
			fileOnDisk: true,
			reserve:    "Qwen 3",
			wantErr:    errDestTaken,
		},
		{
			name:       "file on disk rejected even for the same name",
			existing:   &catalogueEntry{Name: "Qwen 3", Capabilities: &gatewaypb.Capabilities{}},
			fileOnDisk: true,
			reserve:    "Qwen 3",
			wantErr:    errDestTaken,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			modelsDir := t.TempDir()
			c := &metadataCache{
				logger:    zerolog.Nop(),
				modelsDir: modelsDir,
				path:      filepath.Join(t.TempDir(), catalogueFilename),
				entries:   map[string]*catalogueEntry{},
			}
			dest := filepath.Join(modelsDir, "gguf", "qwen-3", "model.gguf")
			if tt.existing != nil {
				c.entries["gguf/qwen-3/model.gguf"] = tt.existing
			}
			if tt.fileOnDisk {
				require.NoError(t, os.MkdirAll(filepath.Dir(dest), 0o755))
				require.NoError(t, os.WriteFile(dest, []byte("x"), 0o644))
			}
			err := c.reserveEntry(dest, tt.reserve)
			if tt.wantErr == nil {
				require.NoError(t, err)
				require.Equal(t, tt.reserve, c.entries["gguf/qwen-3/model.gguf"].Name)
				return
			}
			require.ErrorIs(t, err, tt.wantErr)
			if tt.existing != nil {
				require.Equal(t, tt.existing.Name, c.entries["gguf/qwen-3/model.gguf"].Name,
					"failed reserve must not overwrite the existing claim")
			}
		})
	}
}

// renameGroup enforces the same slug identity rules: the new name must
// slug to something, and must not collide with a different group. A
// same-slug rename of the group itself (casing tweak) is allowed.
func TestRenameGroup_SlugUniqueness(t *testing.T) {
	newCache := func(t *testing.T) *metadataCache {
		modelsDir := t.TempDir()
		require.NoError(t, os.MkdirAll(filepath.Join(modelsDir, "gguf", "qwen-3"), 0o755))
		require.NoError(t, os.WriteFile(filepath.Join(modelsDir, "gguf", "qwen-3", "a.gguf"), []byte("x"), 0o644))
		return &metadataCache{
			logger:    zerolog.Nop(),
			modelsDir: modelsDir,
			entries: map[string]*catalogueEntry{
				"gguf/qwen-3/a.gguf": {Name: "Qwen 3", Capabilities: &gatewaypb.Capabilities{}},
				"gguf/llama/b.gguf":  {Name: "Llama", Capabilities: &gatewaypb.Capabilities{}},
			},
		}
	}

	t.Run("collision with another group rejected", func(t *testing.T) {
		c := newCache(t)
		_, err := c.renameGroup("Qwen 3", "llama")
		require.ErrorIs(t, err, errSlugCollision)
	})
	t.Run("non-sluggable new name rejected", func(t *testing.T) {
		c := newCache(t)
		_, err := c.renameGroup("Qwen 3", "北京")
		require.ErrorIs(t, err, errNameNotSluggable)
	})
	t.Run("same-slug self-rename allowed", func(t *testing.T) {
		c := newCache(t)
		n, err := c.renameGroup("Qwen 3", "qwen 3")
		require.NoError(t, err)
		require.Equal(t, 1, n)
		require.Equal(t, "qwen 3", c.entries["gguf/qwen-3/a.gguf"].Name)
	})
}

// Stale reservations must not leak forever: a reservation older than
// reservationTTL whose file never landed is swept by pruneAbsent. A
// fresh reservation, or an expired one whose file actually exists,
// survives.
func TestPruneAbsent_ReservationTTL(t *testing.T) {
	modelsDir := t.TempDir()
	require.NoError(t, os.MkdirAll(filepath.Join(modelsDir, "m"), 0o755))
	require.NoError(t, os.WriteFile(filepath.Join(modelsDir, "m", "landed.gguf"), []byte("x"), 0o644))

	expired := time.Now().Add(-reservationTTL - time.Hour)
	tests := []struct {
		name     string
		key      string
		entry    *catalogueEntry
		wantKept bool
	}{
		{
			name:     "fresh reservation kept",
			key:      "m/fresh.gguf",
			entry:    &catalogueEntry{Name: "Fresh", ReservedAt: time.Now()},
			wantKept: true,
		},
		{
			name:     "expired reservation without file dropped",
			key:      "m/never-landed.gguf",
			entry:    &catalogueEntry{Name: "Stale", ReservedAt: expired},
			wantKept: false,
		},
		{
			name:     "expired reservation whose file landed kept",
			key:      "m/landed.gguf",
			entry:    &catalogueEntry{Name: "Landed", ReservedAt: expired},
			wantKept: true,
		},
		{
			name:     "legacy reservation without timestamp kept",
			key:      "m/legacy.gguf",
			entry:    &catalogueEntry{Name: "Legacy"},
			wantKept: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := &metadataCache{
				logger:    zerolog.Nop(),
				modelsDir: modelsDir,
				entries:   map[string]*catalogueEntry{tt.key: tt.entry},
			}
			c.pruneAbsent(map[string]struct{}{})
			_, kept := c.entries[tt.key]
			require.Equal(t, tt.wantKept, kept)
		})
	}
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
