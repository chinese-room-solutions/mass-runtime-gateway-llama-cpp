package gateway

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/KernelPryanic/ctxerr"
	gatewaypb "github.com/chinese-room-solutions/mass-proto/gen/go/gateway"
	"github.com/chinese-room-solutions/mass-runtime-llama-cpp/internal/gguf"
	"github.com/rs/zerolog"
)

// catalogueFilename is the gateway-private catalogue file under dataDir.
const catalogueFilename = "models-catalogue.json"

// formatDir is the single subdirectory under modelsDir this gateway
// owns. Every file the walker reads, every absolute path the gateway
// builds for a store-relative ID, every destination computed during
// install lands inside it.
const formatDir = "gguf"

// formatRoot returns the absolute directory holding all files this
// gateway owns: <modelsDir>/<formatDir>/. The single source of truth
// for "where on disk does my catalogue live" — open-coding this join
// is how layout drift bugs creep in.
func formatRoot(modelsDir string) string {
	return filepath.Join(modelsDir, formatDir)
}

// absForStoreID resolves a forward-slashed store-relative ID (the
// Model.id MASS holds, e.g. "qwen3-5-4b/Qwen3.5-4B-UD-Q4_K_XL.gguf")
// to its absolute on-disk path. Use this anywhere a store-relative ID
// has to become a real path; never re-derive by hand.
func absForStoreID(modelsDir, storeID string) string {
	return filepath.Join(formatRoot(modelsDir), filepath.FromSlash(storeID))
}

// errEntryReserved signals that a rename can't proceed because at
// least one matching entry hasn't been committed yet (download in
// flight). Callers translate this to gRPC FailedPrecondition.
var errEntryReserved = errors.New("entry has no header data yet (download in flight); try again when complete")

// metadataCache is the gateway's authoritative catalogue of installed
// models. Each entry is one file on disk; entries are grouped into
// Models by exact match on the operator-supplied Name.
//
// Identity is the operator-typed Name. The gateway never derives it
// from filenames or headers; the install dialog asks for it and
// PlanModelFiles / PlanLocalImport stamp it on every file in the
// bundle. Same Name → same Model. Renames are an explicit RPC.
//
// Walk-time fields (VariantLabel, Companion, ModelType, Capabilities,
// Properties, MTime, Size) are filled in on first observation by
// reading the GGUF header, and refreshed when (mtime, size) drift.
type metadataCache struct {
	logger zerolog.Logger
	path   string

	mu      sync.Mutex
	entries map[string]*catalogueEntry // absPath → entry
	dirty   bool
}

// catalogueEntry is one persisted model file. Name is the only
// identity; everything else is header-derived display data.
type catalogueEntry struct {
	Name         string                  `json:"name"`
	VariantLabel string                  `json:"variant_label,omitempty"`
	Companion    string                  `json:"companion,omitempty"` // "mmproj" or ""
	MTime        time.Time               `json:"mtime,omitzero"`
	Size         int64                   `json:"size,omitempty"`
	ModelType    string                  `json:"model_type,omitempty"`
	Capabilities *gatewaypb.Capabilities `json:"capabilities,omitempty"`
	Properties   map[string]string       `json:"properties,omitempty"`
}

// catalogueFileSchema is the on-disk shape. JSON parse errors clear
// the catalogue (rebuild from scratch); fields that no longer exist
// in catalogueEntry are silently dropped by the JSON decoder.
type catalogueFileSchema struct {
	Entries map[string]*catalogueEntry `json:"entries"`
}

// newMetadataCache constructs a catalogue rooted at dataDir. The disk
// file is loaded if present; missing or unreadable files start empty.
func newMetadataCache(dataDir string, logger zerolog.Logger) *metadataCache {
	c := &metadataCache{
		logger:  logger.With().Str("component", "metadata-catalogue").Logger(),
		path:    filepath.Join(dataDir, catalogueFilename),
		entries: map[string]*catalogueEntry{},
	}
	c.loadFromDisk()
	return c
}

// loadFromDisk populates entries from the catalogue file. Missing
// file is fine; parse errors clear the catalogue (rebuild from
// scratch). Entries without a Name are dropped on load — a missing
// Name means the entry was never properly installed.
func (c *metadataCache) loadFromDisk() {
	buf, err := os.ReadFile(c.path)
	if err != nil {
		if !errors.Is(err, os.ErrNotExist) {
			c.logger.Warn().Err(err).Str("path", c.path).Msg("reading catalogue file; starting empty")
		}
		return
	}
	var schema catalogueFileSchema
	if err := json.Unmarshal(buf, &schema); err != nil {
		c.logger.Warn().Err(err).Str("path", c.path).Msg("parsing catalogue file; discarding")
		return
	}
	for k, v := range schema.Entries {
		if v == nil || strings.TrimSpace(v.Name) == "" {
			continue
		}
		c.entries[k] = v
	}
	c.logger.Debug().Int("entries", len(c.entries)).Msg("loaded model catalogue")
}

// saveToDisk persists the current map. Atomic write (temp + rename)
// so a crash mid-write doesn't leave a half-written file. No-op when
// clean.
func (c *metadataCache) saveToDisk() {
	c.mu.Lock()
	if !c.dirty {
		c.mu.Unlock()
		return
	}
	schema := catalogueFileSchema{
		Entries: make(map[string]*catalogueEntry, len(c.entries)),
	}
	for k, v := range c.entries {
		schema.Entries[k] = v
	}
	c.dirty = false
	c.mu.Unlock()

	buf, err := json.MarshalIndent(schema, "", "  ")
	if err != nil {
		c.logger.Warn().Err(err).Msg("marshalling catalogue; skipping save")
		c.markDirty()
		return
	}
	tmp := c.path + ".tmp"
	if err := os.WriteFile(tmp, buf, 0o644); err != nil {
		c.logger.Warn().Err(err).Str("path", tmp).Msg("writing catalogue file; skipping save")
		c.markDirty()
		return
	}
	if err := os.Rename(tmp, c.path); err != nil {
		c.logger.Warn().Err(err).Str("path", c.path).Msg("renaming catalogue file; skipping save")
		_ = os.Remove(tmp)
		c.markDirty()
		return
	}
}

func (c *metadataCache) markDirty() {
	c.mu.Lock()
	c.dirty = true
	c.mu.Unlock()
}

// reserveEntry pre-writes the operator-supplied Name for one file.
// Called by PlanModelFiles / PlanLocalImport before bytes land on
// disk. Walk-time fields stay zero until the first walk fills them
// in; (mtime, size) staleness later triggers refresh.
func (c *metadataCache) reserveEntry(absPath, name string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	entry, ok := c.entries[absPath]
	if !ok {
		entry = &catalogueEntry{}
		c.entries[absPath] = entry
	}
	entry.Name = name
	c.dirty = true
}

// renameGroup rewrites every catalogue entry whose current Name
// matches old to use new, and moves each affected file on disk into
// the new group's subdirectory under modelsDir/gguf/. Returns the
// number of entries touched.
//
// Refuses (errEntryReserved) when any matching entry is reserved (no
// header read yet) — moving an in-flight download mid-copy would
// orphan its temp file. Callers should retry once the download
// completes.
//
// Multi-file moves aren't atomic at the OS level. If a rename of one
// file fails after others have already moved, the partial state is
// reflected in the catalogue (each successful move updates its entry
// and key) and the error is returned. Operator can retry; subsequent
// calls are idempotent for files already at the new path.
func (c *metadataCache) renameGroup(modelsDir, old, newName string) (int, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if old == "" || newName == "" || old == newName {
		return 0, nil
	}
	// Pre-flight: collect matching paths and refuse if any entry is
	// reserved (no header data yet).
	var matchPaths []string
	for path, e := range c.entries {
		if e == nil || e.Name != old {
			continue
		}
		if e.Capabilities == nil {
			return 0, errEntryReserved
		}
		matchPaths = append(matchPaths, path)
	}
	newSlug := modelSlug(newName)
	root := formatRoot(modelsDir)
	movedKeys := make(map[string]string) // oldPath → newPath
	for _, path := range matchPaths {
		e := c.entries[path]
		newPath := filepath.Join(root, newSlug, filepath.Base(path))
		if newPath == path {
			// Idempotent: file already at destination.
			e.Name = newName
			continue
		}
		if err := os.MkdirAll(filepath.Dir(newPath), 0o755); err != nil {
			return len(movedKeys), ctxerr.With(fmt.Errorf("creating destination dir: %w", err), map[string]any{"path": filepath.Dir(newPath)})
		}
		if err := os.Rename(path, newPath); err != nil {
			return len(movedKeys), ctxerr.With(fmt.Errorf("renaming %s → %s: %w", path, newPath, err), map[string]any{"old": path, "new": newPath})
		}
		movedKeys[path] = newPath
		e.Name = newName
	}
	for oldPath, newPath := range movedKeys {
		c.entries[newPath] = c.entries[oldPath]
		delete(c.entries, oldPath)
	}
	// Best-effort: drop the now-empty old slug folder so the disk
	// matches the catalogue. Ignore errors (folder might still hold
	// unrelated files or have already been cleaned up).
	_ = os.Remove(filepath.Join(root, modelSlug(old)))
	if len(matchPaths) > 0 {
		c.dirty = true
	}
	return len(matchPaths), nil
}

// nameForSlug looks up the source Name whose slug matches s. Used by
// RenameGroup to resolve the slugged Group.id MASS holds back to the
// stored Name. Returns "" when no entry's slug matches.
func (c *metadataCache) nameForSlug(s string) string {
	c.mu.Lock()
	defer c.mu.Unlock()
	for _, e := range c.entries {
		if e == nil || e.Name == "" {
			continue
		}
		if modelSlug(e.Name) == s {
			return e.Name
		}
	}
	return ""
}

// distinctNames returns every Name currently in the catalogue, sorted
// case-insensitively. Used by the install-dialog autocomplete.
func (c *metadataCache) distinctNames() []string {
	c.mu.Lock()
	defer c.mu.Unlock()
	seen := map[string]struct{}{}
	for _, e := range c.entries {
		if e == nil || e.Name == "" {
			continue
		}
		seen[e.Name] = struct{}{}
	}
	out := make([]string, 0, len(seen))
	for n := range seen {
		out = append(out, n)
	}
	return out
}

// pruneAbsent drops every committed entry whose file is no longer in
// observed. Reserved entries — those without header data, pre-written
// by PlanLocalImport / PlanModelFiles before the file lands on disk —
// are kept so a download in flight (or an in-progress local copy)
// doesn't lose its reservation across walks. "Committed" means
// Capabilities has been populated from a successful header read.
func (c *metadataCache) pruneAbsent(observed map[string]struct{}) {
	c.mu.Lock()
	defer c.mu.Unlock()
	for k, v := range c.entries {
		if _, ok := observed[k]; ok {
			continue
		}
		if v == nil || v.Capabilities == nil {
			// Reserved (no header read yet) — leave it for the file
			// to land or for an explicit operator delete.
			continue
		}
		delete(c.entries, k)
		c.dirty = true
	}
}

// walkAndParseModels enumerates every GGUF file under root, returning
// each as a fully-decorated parsedModel. Layout is one model-slug
// subdirectory per Model containing the GGUF files; the walk recurses
// one level into those subdirectories. Files without a catalogue
// entry are logged and skipped — they weren't installed via the
// supported flow. Cache-warm reads reuse the catalogue's stored
// fields; cold reads (or stale (mtime,size) pairs) open the GGUF and
// refresh header-derived data. After the walk, entries whose file no
// longer exists are pruned and the catalogue is persisted if anything
// changed.
func (c *metadataCache) walkAndParseModels(root string) ([]*parsedModel, error) {
	entries, err := os.ReadDir(root)
	if err != nil {
		if os.IsNotExist(err) {
			c.pruneAbsent(map[string]struct{}{})
			c.saveToDisk()
			return nil, nil
		}
		return nil, err
	}
	observed := map[string]struct{}{}
	var out []*parsedModel
	for _, d := range entries {
		if d.IsDir() {
			subRoot := filepath.Join(root, d.Name())
			subEntries, err := os.ReadDir(subRoot)
			if err != nil {
				c.logger.Warn().Err(err).Str("path", subRoot).Msg("reading model subdirectory; skipping")
				continue
			}
			for _, s := range subEntries {
				if s.IsDir() {
					continue
				}
				// storeID is relative to <root> (gguf/) so the
				// detail handler can rebuild the absolute path
				// from the URL-encoded id alone.
				storeID := d.Name() + "/" + s.Name()
				if info, ok := c.maybeParseFileWithID(subRoot, s.Name(), storeID); ok {
					observed[info.AbsolutePath] = struct{}{}
					out = append(out, info)
				}
			}
			continue
		}
		if info, ok := c.maybeParseFile(root, d.Name()); ok {
			observed[info.AbsolutePath] = struct{}{}
			out = append(out, info)
		}
	}
	c.pruneAbsent(observed)
	c.saveToDisk()
	return out, nil
}

// maybeParseFile filters one directory entry by extension + temp-file
// convention and forwards to parseModelInfo. Returns ok=false when
// the entry isn't a GGUF the gateway should consider.
func (c *metadataCache) maybeParseFile(dir, name string) (*parsedModel, bool) {
	return c.maybeParseFileWithID(dir, name, name)
}

// maybeParseFileWithID is the same as maybeParseFile but lets the
// caller stamp a storeID different from the filename — used when
// files live in per-model subdirectories under gguf/, where the id
// must include the subdirectory so the detail handler can rebuild
// the absolute path from the id alone.
func (c *metadataCache) maybeParseFileWithID(dir, name, storeID string) (*parsedModel, bool) {
	if !strings.EqualFold(extOf(name), "gguf") {
		return nil, false
	}
	// Skip MASS's in-progress download temp files. Format produced by
	// pkg/download.TempFilePath is ".downloading-<basename>".
	if strings.HasPrefix(name, ".downloading-") {
		return nil, false
	}
	return c.parseModelInfo(filepath.Join(dir, name), storeID)
}

// parseModelInfo returns the catalogue's view of one file. Files
// without a catalogue entry are logged and skipped — they weren't
// installed via PlanModelFiles / PlanLocalImport. Header is read on
// first observation or when (mtime, size) has drifted.
func (c *metadataCache) parseModelInfo(absPath, storeID string) (*parsedModel, bool) {
	st, err := os.Stat(absPath)
	if err != nil {
		return nil, false
	}
	mtime, size := st.ModTime(), st.Size()

	c.mu.Lock()
	entry, hasEntry := c.entries[absPath]
	c.mu.Unlock()
	if !hasEntry {
		c.logger.Warn().Str("path", absPath).Msg("file present without catalogue entry; ignoring (use Browse Local or HF install to register it)")
		return nil, false
	}

	stale := !entry.MTime.Equal(mtime) || entry.Size != size
	needsHeader := stale || entry.Capabilities == nil

	if needsHeader {
		meta, err := gguf.ReadMeta(absPath)
		if err != nil {
			c.logger.Debug().Err(err).Str("path", absPath).Msg("reading GGUF header; skipping")
			return nil, false
		}
		kv := meta.Summary()
		c.mu.Lock()
		entry.VariantLabel = kv["quant"]
		entry.Companion = ""
		if strings.EqualFold(kv["architecture"], "clip") {
			entry.Companion = "mmproj"
			entry.VariantLabel = "Mmproj"
		}
		entry.Capabilities = capabilitiesFromHeader(kv)
		entry.ModelType = modelTypeFromHeader(kv)
		entry.Properties = propertiesFromKV(kv)
		entry.MTime = mtime
		entry.Size = size
		c.dirty = true
		c.mu.Unlock()
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	// Per-file capabilities: a projector is part of the vision
	// pipeline, not a standalone capable file — its row carries the
	// "Mmproj" badge instead of capability icons. A non-companion
	// file claims vision when a same-Name mmproj sibling exists,
	// re-evaluated every call since siblings can land/leave after
	// the entry was written.
	caps := cloneCapabilities(entry.Capabilities)
	if caps == nil {
		caps = &gatewaypb.Capabilities{}
	}
	if entry.Companion == "mmproj" {
		caps = &gatewaypb.Capabilities{}
	} else if entry.Name != "" {
		for path, other := range c.entries {
			if path == absPath || other == nil {
				continue
			}
			if other.Name == entry.Name && other.Companion == "mmproj" {
				caps.Vision = true
				break
			}
		}
	}
	props := make(map[string]string, len(entry.Properties))
	for k, v := range entry.Properties {
		props[k] = v
	}
	return &parsedModel{
		ID:           storeID,
		AbsolutePath: absPath,
		SizeBytes:    size,
		Name:         entry.Name,
		VariantLabel: entry.VariantLabel,
		Companion:    entry.Companion,
		ModelType:    entry.ModelType,
		Capabilities: caps,
		Properties:   props,
	}, true
}

func cloneCapabilities(in *gatewaypb.Capabilities) *gatewaypb.Capabilities {
	if in == nil {
		return nil
	}
	return &gatewaypb.Capabilities{
		Vision:   in.GetVision(),
		Audio:    in.GetAudio(),
		Thinking: in.GetThinking(),
	}
}
