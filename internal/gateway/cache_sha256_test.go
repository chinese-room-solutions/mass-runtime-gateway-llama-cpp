package gateway

import (
	"os"
	"path/filepath"
	"testing"

	gatewaypb "github.com/chinese-room-solutions/mass-proto/gen/go/gateway"
	"github.com/rs/zerolog"
	"github.com/stretchr/testify/require"
)

// hashFile streams a file through SHA-256 and returns the lowercase-hex
// digest. A missing file is an error the caller degrades on (empty hash
// → worker skips verification), never a panic.
func TestHashFile(t *testing.T) {
	tmp := t.TempDir()
	f := filepath.Join(tmp, "content.bin")
	require.NoError(t, os.WriteFile(f, []byte("hello world"), 0o644))

	tests := []struct {
		name    string
		path    string
		want    string
		wantErr bool
	}{
		{
			name: "known content → known digest",
			path: f,
			// echo -n 'hello world' | sha256sum
			want: "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9",
		},
		{
			name:    "missing file → error",
			path:    filepath.Join(tmp, "nope.bin"),
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := hashFile(tt.path)
			if tt.wantErr {
				require.Error(t, err)
				require.Empty(t, got)
				return
			}
			require.NoError(t, err)
			require.Equal(t, tt.want, got)
		})
	}
}

// parseModelInfo backfills Sha256 on a committed entry whose header data
// is already good (needsHeader=false) — the hash branch fires on its own
// trigger and does not force a GGUF re-parse, so a non-GGUF file body is
// fine here. This is the walk-level "entry gains Sha256 on first commit"
// case reachable without an on-disk GGUF builder.
func TestParseModelInfo_BackfillsSha256WithoutHeaderReparse(t *testing.T) {
	tmp := t.TempDir()
	c := newMetadataCache(tmp, tmp, zerolog.Nop())

	abs := filepath.Join(tmp, "model.gguf")
	body := []byte("hello world")
	require.NoError(t, os.WriteFile(abs, body, 0o644))
	st, err := os.Stat(abs)
	require.NoError(t, err)

	// Committed, non-stale entry: mtime/size match, capabilities and
	// both parameter counts present → needsHeader is false. Only the
	// empty Sha256 should trigger work.
	c.entries["model.gguf"] = &catalogueEntry{
		Name:                 "m",
		Capabilities:         &gatewaypb.Capabilities{},
		ParameterCount:       8_000_000_000,
		ActiveParameterCount: 8_000_000_000,
		MTime:                st.ModTime(),
		Size:                 st.Size(),
	}

	_, ok := c.parseModelInfo(abs, "model.gguf")
	require.True(t, ok)

	const want = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
	require.Equal(t, want, c.entries["model.gguf"].Sha256, "first commit must backfill the digest")
	// Header data untouched (no re-parse forced by the missing hash).
	require.Equal(t, uint64(8_000_000_000), c.entries["model.gguf"].ParameterCount)

	// The walk persists after parsing; do the same and confirm the
	// digest round-trips to disk.
	c.saveToDisk()
	persisted := readCatalogueFile(t, tmp)
	require.Equal(t, want, persisted["model.gguf"].Sha256)
}

// sha256For returns the persisted digest fast-path without touching
// disk, mirroring parameterCount's branch structure. The lazy-backfill
// path (absent digest, real file) is covered by
// TestParseModelInfo_BackfillsSha256WithoutHeaderReparse; here we assert
// the pure branches.
func TestMetadataCacheSha256For_Branches(t *testing.T) {
	const digest = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
	tmp := t.TempDir()
	tests := []struct {
		name  string
		entry *catalogueEntry // nil = no entry at that path
		want  string
	}{
		{
			name:  "absent path returns empty",
			entry: nil,
			want:  "",
		},
		{
			name:  "persisted digest returned verbatim",
			entry: &catalogueEntry{Name: "m", Sha256: digest},
			want:  digest,
		},
		{
			// Entry present but digest empty and no file on disk — the
			// backfill parseModelInfo fails quietly and we return "".
			name:  "empty digest without on-disk file → empty",
			entry: &catalogueEntry{Name: "m"},
			want:  "",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := newMetadataCache(tmp, tmp, zerolog.Nop())
			absPath := filepath.Join(tmp, "model.gguf")
			if tt.entry != nil {
				c.entries["model.gguf"] = tt.entry
			}
			require.Equal(t, tt.want, c.sha256For(absPath))
		})
	}
}
