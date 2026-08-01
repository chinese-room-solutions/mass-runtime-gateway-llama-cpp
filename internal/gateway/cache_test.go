package gateway

import (
	"path/filepath"
	"testing"

	llamacpp "github.com/chinese-room-solutions/mass-proto/gen/go/llama-cpp"
	workerpb "github.com/chinese-room-solutions/mass-proto/gen/go/worker"
	"github.com/rs/zerolog"
	"github.com/stretchr/testify/require"
)

// parameterCount returns the persisted count fast-path without
// touching disk. The lazy-backfill path requires a real GGUF on disk
// and is exercised in dispatch integration tests; this unit table
// focuses on the three pure branches.
func TestMetadataCacheParameterCount_Branches(t *testing.T) {
	tests := []struct {
		name  string
		entry *catalogueEntry // nil = no entry at that path
		want  uint64
	}{
		{
			// Missing entry — caller falls back to fallbackParameterCount.
			name:  "absent path returns 0",
			entry: nil,
			want:  0,
		},
		{
			// Hot path: catalogue already has a count. parameterCount
			// returns it without going to disk.
			name:  "persisted count returned verbatim",
			entry: &catalogueEntry{Name: "m1", ParameterCount: 8_030_261_248},
			want:  8_030_261_248,
		},
		{
			// Entry exists but ParameterCount is 0 — the lazy-backfill
			// path tries to re-read the GGUF header. Since the file
			// doesn't exist on disk in this test, parseModelInfo fails
			// quietly and we return 0 — same fallback semantics as
			// "absent path".
			name:  "stale entry without on-disk file → 0",
			entry: &catalogueEntry{Name: "m1", ParameterCount: 0},
			want:  0,
		},
	}

	tmp := t.TempDir()
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := newMetadataCache(tmp, tmp, zerolog.Nop())
			absPath := filepath.Join(tmp, "model.gguf")
			if tt.entry != nil {
				c.entries["model.gguf"] = tt.entry
			}
			require.Equal(t, tt.want, c.parameterCount(absPath))
		})
	}
}

// primaryParameterCount picks the file with role PRIMARY out of a
// dispatch's Files slice and forwards its local_path to the metadata
// cache. Mmproj companions don't count; missing local_path doesn't
// count; nil cache short-circuits to 0 (defensive — handlers without
// catalogue plumbing should not crash).
func TestPrimaryParameterCount_Branches(t *testing.T) {
	tmp := t.TempDir()
	cache := newMetadataCache(tmp, tmp, zerolog.Nop())
	primaryPath := filepath.Join(tmp, "gguf", "m1", "primary.gguf")
	cache.entries["gguf/m1/primary.gguf"] = &catalogueEntry{Name: "m1", ParameterCount: 5_500_000_000}

	tests := []struct {
		name  string
		h     *handlers
		files []*workerpb.ModelFile
		want  uint64
	}{
		{
			// Defensive: handler without cache (early-init paths)
			// returns 0 rather than NPE.
			name:  "nil handler returns 0",
			h:     nil,
			files: []*workerpb.ModelFile{{Role: workerpb.ModelFileRole_MODEL_FILE_ROLE_PRIMARY, LocalPath: primaryPath}},
			want:  0,
		},
		{
			// Cache unset: scheduler is wired but catalogue isn't.
			name:  "handler without cache returns 0",
			h:     &handlers{},
			files: []*workerpb.ModelFile{{Role: workerpb.ModelFileRole_MODEL_FILE_ROLE_PRIMARY, LocalPath: primaryPath}},
			want:  0,
		},
		{
			// Happy path: a single primary file with a populated cache.
			name:  "primary file resolves to cached count",
			h:     &handlers{cache: cache},
			files: []*workerpb.ModelFile{{Role: workerpb.ModelFileRole_MODEL_FILE_ROLE_PRIMARY, LocalPath: primaryPath}},
			want:  5_500_000_000,
		},
		{
			// Companion (mmproj) ahead of primary in the slice — we
			// must keep walking until we find PRIMARY, not stop at
			// the first non-nil entry.
			name: "mmproj before primary is skipped",
			h:    &handlers{cache: cache},
			files: []*workerpb.ModelFile{
				{Role: workerpb.ModelFileRole_MODEL_FILE_ROLE_MMPROJ, LocalPath: "/abs/proj.gguf"},
				{Role: workerpb.ModelFileRole_MODEL_FILE_ROLE_PRIMARY, LocalPath: primaryPath},
			},
			want: 5_500_000_000,
		},
		{
			// nil entry in the slice — defensive: we keep walking
			// instead of NPEing on f.GetRole().
			name: "nil file entry is skipped",
			h:    &handlers{cache: cache},
			files: []*workerpb.ModelFile{
				nil,
				{Role: workerpb.ModelFileRole_MODEL_FILE_ROLE_PRIMARY, LocalPath: primaryPath},
			},
			want: 5_500_000_000,
		},
		{
			// Primary file present but local_path empty (download
			// hasn't completed yet). We can't dereference the
			// catalogue without the path, so fall back to 0.
			name:  "primary without local_path returns 0",
			h:     &handlers{cache: cache},
			files: []*workerpb.ModelFile{{Role: workerpb.ModelFileRole_MODEL_FILE_ROLE_PRIMARY, LocalPath: ""}},
			want:  0,
		},
		{
			// No primary in the slice at all — companion-only dispatch
			// (shouldn't happen, defensive).
			name:  "no primary returns 0",
			h:     &handlers{cache: cache},
			files: []*workerpb.ModelFile{{Role: workerpb.ModelFileRole_MODEL_FILE_ROLE_MMPROJ, LocalPath: "/abs/proj.gguf"}},
			want:  0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			require.Equal(t, tt.want, tt.h.primaryParameterCount(tt.files))
		})
	}
}

// metadataCache.properties returns a snapshot of an entry's Properties
// map without holding the cache mutex past return — used by the
// gateway-side load-bytes estimator to read layers/embedding/etc.
// Empty / absent entries return nil so the estimator can detect
// missing metadata cleanly.
func TestMetadataCacheProperties(t *testing.T) {
	tmp := t.TempDir()
	tests := []struct {
		name  string
		entry *catalogueEntry
		want  map[string]string
	}{
		{
			name:  "absent path returns nil",
			entry: nil,
			want:  nil,
		},
		{
			name:  "entry with nil Properties returns nil",
			entry: &catalogueEntry{Name: "m"},
			want:  nil,
		},
		{
			name:  "entry with empty Properties returns nil",
			entry: &catalogueEntry{Name: "m", Properties: map[string]string{}},
			want:  nil,
		},
		{
			name: "entry with Properties returns a copy",
			entry: &catalogueEntry{
				Name:       "m",
				Properties: map[string]string{"layers": "32", "embedding": "4096"},
			},
			want: map[string]string{"layers": "32", "embedding": "4096"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := newMetadataCache(tmp, tmp, zerolog.Nop())
			absPath := filepath.Join(tmp, "model.gguf")
			if tt.entry != nil {
				c.entries["model.gguf"] = tt.entry
			}
			got := c.properties(absPath)
			require.Equal(t, tt.want, got)
			// Snapshot guarantee: mutating the result must not affect
			// the cache's stored map.
			if got != nil {
				got["intruder"] = "value"
				require.NotContains(t, c.entries["model.gguf"].Properties, "intruder")
			}
		})
	}
}

// handlers.loadByteEstimate returns (base, perSlot, headroom). base
// is weights + scratch; perSlot is one context's KV cache; headroom
// is the explicit LoadHints override or 0 (no override — MASS then
// reads the worker-reported flag). The table covers the
// file-aggregation and catalogue-resolution paths.
func TestHandlersLoadByteEstimate(t *testing.T) {
	const gb = int64(1024 * 1024 * 1024)
	const scratch = int64(512 * 1024 * 1024)
	// 32 layers * 4096 ctx * 32 kv-heads * (4096/32) head_dim * 2 K+V *
	// 2 bytes = 2 GB. Matches TestEstimateLoadBytes.
	const perSlotKV = 2 * gb
	tmp := t.TempDir()
	cache := newMetadataCache(tmp, tmp, zerolog.Nop())
	primaryPath := filepath.Join(tmp, "gguf", "m", "primary.gguf")
	cache.entries["gguf/m/primary.gguf"] = &catalogueEntry{
		Name: "m",
		Properties: map[string]string{
			"layers": "32", "embedding": "4096",
			"head_count": "32", "head_count_kv": "32",
		},
	}

	hintsDefault := &llamacpp.LoadHints{ContextSize: 4096}
	override := int32(60)
	hintsOverride := &llamacpp.LoadHints{ContextSize: 4096, VramHeadroomPct: &override}
	tests := []struct {
		name         string
		h            *handlers
		files        []*workerpb.ModelFile
		hints        *llamacpp.LoadHints
		wantBase     int64
		wantPerSlot  int64
		wantHeadroom int32
	}{
		{
			name:  "nil handler returns zeros",
			h:     nil,
			files: []*workerpb.ModelFile{{Role: workerpb.ModelFileRole_MODEL_FILE_ROLE_PRIMARY, LocalPath: primaryPath, SizeBytes: 5 * gb}},
			hints: hintsDefault,
		},
		{
			name:  "handler without cache returns zeros",
			h:     &handlers{},
			files: []*workerpb.ModelFile{{Role: workerpb.ModelFileRole_MODEL_FILE_ROLE_PRIMARY, LocalPath: primaryPath, SizeBytes: 5 * gb}},
			hints: hintsDefault,
		},
		{
			name:         "primary only: base=weights+scratch, perSlot=KV, no headroom override",
			h:            &handlers{cache: cache},
			files:        []*workerpb.ModelFile{{Role: workerpb.ModelFileRole_MODEL_FILE_ROLE_PRIMARY, LocalPath: primaryPath, SizeBytes: 5 * gb}},
			hints:        hintsDefault,
			wantBase:     5*gb + scratch,
			wantPerSlot:  perSlotKV,
		},
		{
			// Mmproj adds to base (loaded onto the device too); perSlot
			// stays the same since KV math reads only the primary's
			// properties.
			name: "mmproj adds to base",
			h:    &handlers{cache: cache},
			files: []*workerpb.ModelFile{
				{Role: workerpb.ModelFileRole_MODEL_FILE_ROLE_PRIMARY, LocalPath: primaryPath, SizeBytes: 5 * gb},
				{Role: workerpb.ModelFileRole_MODEL_FILE_ROLE_MMPROJ, LocalPath: "/abs/proj.gguf", SizeBytes: 1 * gb},
			},
			hints:        hintsDefault,
			wantBase:     6*gb + scratch,
			wantPerSlot:  perSlotKV,
		},
		{
			name:         "LoadHints headroom override propagates",
			h:            &handlers{cache: cache},
			files:        []*workerpb.ModelFile{{Role: workerpb.ModelFileRole_MODEL_FILE_ROLE_PRIMARY, LocalPath: primaryPath, SizeBytes: 5 * gb}},
			hints:        hintsOverride,
			wantBase:     5*gb + scratch,
			wantPerSlot:  perSlotKV,
			wantHeadroom: 60,
		},
		{
			name:  "zero file bytes returns zeros",
			h:     &handlers{cache: cache},
			files: []*workerpb.ModelFile{{Role: workerpb.ModelFileRole_MODEL_FILE_ROLE_PRIMARY, LocalPath: primaryPath, SizeBytes: 0}},
			hints: hintsDefault,
		},
		{
			name:  "primary without local_path returns zeros",
			h:     &handlers{cache: cache},
			files: []*workerpb.ModelFile{{Role: workerpb.ModelFileRole_MODEL_FILE_ROLE_PRIMARY, LocalPath: "", SizeBytes: 5 * gb}},
			hints: hintsDefault,
		},
		{
			name: "nil file entry is skipped",
			h:    &handlers{cache: cache},
			files: []*workerpb.ModelFile{
				nil,
				{Role: workerpb.ModelFileRole_MODEL_FILE_ROLE_PRIMARY, LocalPath: primaryPath, SizeBytes: 5 * gb},
			},
			hints:        hintsDefault,
			wantBase:     5*gb + scratch,
			wantPerSlot:  perSlotKV,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			base, perSlot, headroom := tt.h.loadByteEstimate(tt.files, tt.hints)
			require.Equal(t, tt.wantBase, base)
			require.Equal(t, tt.wantPerSlot, perSlot)
			require.Equal(t, tt.wantHeadroom, headroom)
		})
	}
}

// visionParamsFor resolves the encoder shape from the companion
// mmproj's catalogue properties; merge is stored as the linear
// spatial_merge_size and squared into patches-per-token. Missing
// pieces (no mmproj on the submit, no cache, absent props) fall back
// to the zero value so the package defaults apply.
func TestHandlersVisionParamsFor(t *testing.T) {
	tmp := t.TempDir()
	cache := newMetadataCache(tmp, tmp, zerolog.Nop())
	mmprojPath := filepath.Join(tmp, "gguf", "m", "mmproj.gguf")
	cache.entries["gguf/m/mmproj.gguf"] = &catalogueEntry{
		Name:      "m",
		Companion: "mmproj",
		Properties: map[string]string{
			"vision_patch_size": "16",
			"vision_merge_size": "2",
		},
	}
	h := &handlers{cache: cache}
	files := []*workerpb.ModelFile{
		{Role: workerpb.ModelFileRole_MODEL_FILE_ROLE_PRIMARY, LocalPath: filepath.Join(tmp, "gguf", "m", "primary.gguf")},
		{Role: workerpb.ModelFileRole_MODEL_FILE_ROLE_MMPROJ, LocalPath: mmprojPath},
	}
	require.Equal(t, visionParams{patchPixels: 16, mergeFactor: 4}, h.visionParamsFor(files))

	// No mmproj on the submit → zero value.
	require.Equal(t, visionParams{}, h.visionParamsFor(files[:1]))
	// Nil cache → zero value.
	require.Equal(t, visionParams{}, (&handlers{}).visionParamsFor(files))
	// Catalogued mmproj without vision props → zero value.
	cache.entries["gguf/m/mmproj.gguf"].Properties = map[string]string{}
	require.Equal(t, visionParams{}, h.visionParamsFor(files))
}
