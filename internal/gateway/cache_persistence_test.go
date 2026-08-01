package gateway

import (
	"encoding/json"
	"os"
	"path/filepath"
	"sync"
	"testing"

	gatewaypb "github.com/chinese-room-solutions/mass-proto/gen/go/gateway"
	"github.com/rs/zerolog"
	"github.com/stretchr/testify/require"
)

// readCatalogueFile parses the raw catalogue JSON written to dataDir so
// tests can assert on the persisted (not in-memory) shape.
func readCatalogueFile(t *testing.T, dataDir string) map[string]*catalogueEntry {
	t.Helper()
	buf, err := os.ReadFile(filepath.Join(dataDir, catalogueFilename))
	require.NoError(t, err)
	var schema catalogueFileSchema
	require.NoError(t, json.Unmarshal(buf, &schema))
	return schema.Entries
}

// writeCatalogueFile writes a raw entries map as the catalogue file.
func writeCatalogueFile(t *testing.T, dataDir string, entries map[string]*catalogueEntry) {
	t.Helper()
	buf, err := json.Marshal(catalogueFileSchema{Entries: entries})
	require.NoError(t, err)
	require.NoError(t, os.WriteFile(filepath.Join(dataDir, catalogueFilename), buf, 0o644))
}

// The catalogue is keyed by modelsDir-relative slash paths so a store
// move (or drive-letter change) doesn't orphan every operator-typed
// Name. Legacy catalogues used absolute keys: on load, keys under the
// current modelsDir migrate to relative, keys outside it are dropped
// loudly, and the migrated file is persisted immediately.
func TestCatalogueLoad_MigratesAbsoluteKeys(t *testing.T) {
	dataDir := t.TempDir()
	modelsDir := t.TempDir()
	elsewhere := t.TempDir()

	insideAbs := filepath.Join(modelsDir, "gguf", "qwen", "a.gguf")
	outsideAbs := filepath.Join(elsewhere, "gguf", "qwen", "stray.gguf")
	writeCatalogueFile(t, dataDir, map[string]*catalogueEntry{
		insideAbs:           {Name: "Qwen"},
		outsideAbs:          {Name: "Stray"},
		"gguf/llama/b.gguf": {Name: "Llama"}, // already relative — untouched
		"gguf/none/x.gguf":  {Name: ""},      // nameless — dropped on load
		"gguf/nil-entry.gg": nil,             // nil — dropped on load
	})

	c := newMetadataCache(dataDir, modelsDir, zerolog.Nop())

	require.Contains(t, c.entries, "gguf/qwen/a.gguf", "absolute key under modelsDir migrates to relative")
	require.Equal(t, "Qwen", c.entries["gguf/qwen/a.gguf"].Name)
	require.Contains(t, c.entries, "gguf/llama/b.gguf")
	require.Len(t, c.entries, 2, "entry outside modelsDir is dropped; nameless/nil entries are dropped")

	// The migration must be persisted immediately (one-time read).
	persisted := readCatalogueFile(t, dataDir)
	require.Contains(t, persisted, "gguf/qwen/a.gguf")
	require.NotContains(t, persisted, insideAbs)
	require.NotContains(t, persisted, outsideAbs)
}

// A catalogue already keyed relatively loads verbatim and does not
// rewrite the file (no spurious migration writes).
func TestCatalogueLoad_RelativeKeysVerbatim(t *testing.T) {
	dataDir := t.TempDir()
	modelsDir := t.TempDir()
	writeCatalogueFile(t, dataDir, map[string]*catalogueEntry{
		"gguf/qwen/a.gguf": {Name: "Qwen", Capabilities: &gatewaypb.Capabilities{}},
	})
	before, err := os.Stat(filepath.Join(dataDir, catalogueFilename))
	require.NoError(t, err)

	c := newMetadataCache(dataDir, modelsDir, zerolog.Nop())
	require.Contains(t, c.entries, "gguf/qwen/a.gguf")
	require.False(t, c.dirty, "clean load must not mark the catalogue dirty")

	after, err := os.Stat(filepath.Join(dataDir, catalogueFilename))
	require.NoError(t, err)
	require.Equal(t, before.ModTime(), after.ModTime(), "clean load must not rewrite the file")
}

// Reservations must survive a crash: reserveEntry persists to disk
// immediately, so a gateway restarted mid-download still knows the
// in-flight file's operator-typed Name.
func TestReserveEntry_PersistsImmediately(t *testing.T) {
	dataDir := t.TempDir()
	modelsDir := t.TempDir()

	c := newMetadataCache(dataDir, modelsDir, zerolog.Nop())
	dest := filepath.Join(modelsDir, "gguf", "qwen-3", "model.gguf")
	require.NoError(t, c.reserveEntry(dest, "Qwen 3"))

	// Simulate a crash: a fresh cache reads only what hit the disk.
	reloaded := newMetadataCache(dataDir, modelsDir, zerolog.Nop())
	entry, ok := reloaded.entries["gguf/qwen-3/model.gguf"]
	require.True(t, ok, "reservation must round-trip through save/load without an explicit save")
	require.Equal(t, "Qwen 3", entry.Name)
	require.False(t, entry.ReservedAt.IsZero(), "reservation timestamp must round-trip for the TTL sweep")
}

// Reservations persisted before ReservedAt existed load with the clock
// stamped at load time, so the TTL sweep can eventually reap them
// instead of leaking forever.
func TestCatalogueLoad_StampsLegacyReservations(t *testing.T) {
	dataDir := t.TempDir()
	modelsDir := t.TempDir()
	writeCatalogueFile(t, dataDir, map[string]*catalogueEntry{
		"gguf/qwen/pending.gguf": {Name: "Qwen"}, // reservation, no reserved_at
	})

	c := newMetadataCache(dataDir, modelsDir, zerolog.Nop())
	entry := c.entries["gguf/qwen/pending.gguf"]
	require.NotNil(t, entry)
	require.False(t, entry.ReservedAt.IsZero(), "legacy reservation must be stamped on load")

	persisted := readCatalogueFile(t, dataDir)
	require.False(t, persisted["gguf/qwen/pending.gguf"].ReservedAt.IsZero(), "stamp must be persisted")
}

// A corrupt catalogue must never be silently discarded — it holds the
// operator-typed Names (the only model identity). It is renamed to
// models-catalogue.json.corrupt-<unixts> and the cache starts empty.
func TestCatalogueLoad_QuarantinesCorruptFile(t *testing.T) {
	dataDir := t.TempDir()
	modelsDir := t.TempDir()
	cataloguePath := filepath.Join(dataDir, catalogueFilename)
	corrupt := []byte(`{"entries": {"gguf/a.gguf": {"name": "Qwen"`) // truncated JSON
	require.NoError(t, os.WriteFile(cataloguePath, corrupt, 0o644))

	c := newMetadataCache(dataDir, modelsDir, zerolog.Nop())
	require.Empty(t, c.entries, "corrupt catalogue starts empty")
	require.NoFileExists(t, cataloguePath, "corrupt file must be moved aside, not left in place")

	matches, err := filepath.Glob(cataloguePath + ".corrupt-*")
	require.NoError(t, err)
	require.Len(t, matches, 1, "exactly one quarantined copy")
	preserved, err := os.ReadFile(matches[0])
	require.NoError(t, err)
	require.Equal(t, corrupt, preserved, "quarantined bytes must be the original file, untouched")
}

// saveToDisk marshals the entries map under the lock: entries are
// pointers mutated in place (parseModelInfo, reserveEntry), so
// serialising a shallow copy outside the lock races those writes.
// This test only proves its worth under `go test -race`.
func TestSaveToDisk_NoRaceWithConcurrentMutation(t *testing.T) {
	dataDir := t.TempDir()
	modelsDir := t.TempDir()
	c := newMetadataCache(dataDir, modelsDir, zerolog.Nop())
	require.NoError(t, c.reserveEntry(filepath.Join(modelsDir, "gguf", "q", "a.gguf"), "Q"))

	var wg sync.WaitGroup
	for range 4 {
		wg.Add(2)
		go func() {
			defer wg.Done()
			for i := range 50 {
				// Mimic parseModelInfo: mutate the shared entry under the lock.
				c.mu.Lock()
				c.entries["gguf/q/a.gguf"].VariantLabel = "Q" + string(rune('0'+i%10))
				c.dirty = true
				c.mu.Unlock()
			}
		}()
		go func() {
			defer wg.Done()
			for range 50 {
				c.saveToDisk()
			}
		}()
	}
	wg.Wait()
}

// entryKey/entryAbs round-trip: absolute paths under modelsDir map to
// slash-relative keys and back; paths outside modelsDir are rejected.
func TestEntryKeyRoundTrip(t *testing.T) {
	modelsDir := t.TempDir()
	c := &metadataCache{modelsDir: modelsDir, entries: map[string]*catalogueEntry{}, logger: zerolog.Nop()}

	tests := []struct {
		name    string
		abs     string
		wantKey string
		wantOK  bool
	}{
		{
			name:    "file under modelsDir",
			abs:     filepath.Join(modelsDir, "gguf", "g", "a.gguf"),
			wantKey: "gguf/g/a.gguf",
			wantOK:  true,
		},
		{
			name:   "file outside modelsDir",
			abs:    filepath.Join(filepath.Dir(modelsDir), "elsewhere", "a.gguf"),
			wantOK: false,
		},
		{
			name:   "modelsDir parent",
			abs:    filepath.Dir(modelsDir),
			wantOK: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			key, ok := c.entryKey(tt.abs)
			require.Equal(t, tt.wantOK, ok)
			if !tt.wantOK {
				return
			}
			require.Equal(t, tt.wantKey, key)
			require.Equal(t, tt.abs, c.entryAbs(key))
		})
	}
}
