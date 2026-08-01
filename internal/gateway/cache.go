package gateway

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path"
	"path/filepath"
	"slices"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/KernelPryanic/ctxerr"
	gatewaypb "github.com/chinese-room-solutions/mass-proto/gen/go/gateway"
	"github.com/chinese-room-solutions/mass-runtime-gateway-llama-cpp/internal/gguf"
	"github.com/rs/zerolog"
)

// catalogueFilename is the gateway-private catalogue file under dataDir.
const catalogueFilename = "models-catalogue.json"

// formatDir is the single subdirectory under modelsDir this gateway
// owns. Every file the walker reads, every absolute path the gateway
// builds for a store-relative ID, every destination computed during
// install lands inside it.
const formatDir = "gguf"

// downloadingPrefix marks MASS's in-progress download temp files:
// MASS's download manager writes ".downloading-<basename>" next to the
// destination and renames it into place on completion. The walk skips
// these — a half-written GGUF must never enter the catalogue. The
// convention is MASS's (its pkg/download.TempFilePath); this constant
// is the gateway's documented mirror of it.
const downloadingPrefix = ".downloading-"

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

// errNameNotSluggable rejects a group name whose slug is empty (no
// ASCII letters or digits at all) — MASS's Group.id would be "" and
// the group unaddressable. Callers translate to InvalidArgument.
var errNameNotSluggable = errors.New("group name not usable")

// errDestTaken rejects a plan whose destination path is already
// occupied — by a file on disk, or by a live catalogue entry for a
// DIFFERENT group name (e.g. a reservation for a download still in
// flight). Callers translate to AlreadyExists.
var errDestTaken = errors.New("destination already taken")

// errSlugCollision rejects a group name whose slug equals a DIFFERENT
// existing Name's slug. Group.id = modelSlug(Name); two names sharing
// a slug make nameForSlug/relPathsForModel resolve arbitrarily — a
// delete could plan the wrong group's files. Callers translate to
// AlreadyExists.
var errSlugCollision = errors.New("group name conflicts with an existing group")

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
//
// Entries are keyed by the file's slash-normalized path RELATIVE to
// modelsDir (e.g. "gguf/qwen-3/model.gguf") so the catalogue survives
// a models-dir move or a drive-letter change. Public methods still
// accept absolute paths and convert via entryKey/entryAbs.
type metadataCache struct {
	logger    zerolog.Logger
	path      string
	modelsDir string

	mu      sync.Mutex
	entries map[string]*catalogueEntry // modelsDir-relative slash key → entry
	dirty   bool
}

// reservationTTL bounds how long a reservation (an entry pre-written
// before its file lands, no header data yet) may sit without the file
// ever appearing. Downloads that die without cleanup would otherwise
// block the destination and pollute the autocomplete forever. The
// sweep runs inside every walk (pruneAbsent).
const reservationTTL = 48 * time.Hour

// catalogueEntry is one persisted model file. Name is the only
// identity; everything else is header-derived display data.
type catalogueEntry struct {
	Name           string    `json:"name"`
	VariantLabel   string    `json:"variant_label,omitempty"`
	Companion      string    `json:"companion,omitempty"`  // "mmproj" or ""
	ReservedAt     time.Time `json:"reserved_at,omitzero"` // set while the entry is a reservation; cleared on first header read
	MTime          time.Time `json:"mtime,omitzero"`
	Size           int64     `json:"size,omitempty"`
	Sha256         string    `json:"sha256,omitempty"`          // full-file lowercase-hex digest for remote-worker integrity; "" = unknown, worker skips verification
	ParameterCount uint64    `json:"parameter_count,omitempty"` // GGUF general.parameter_count, tensor-table sum, or parsed size_label; 0 = unknown
	// ActiveParameterCount is the per-token active parameter count that
	// compute-cost prediction scales with — smaller than ParameterCount
	// for MoE models, equal for dense ones. 0 = not yet computed (entry
	// persisted before this field existed, or the header carried no
	// usable count); parseModelInfo backfills on the next header read.
	ActiveParameterCount uint64                  `json:"active_parameter_count,omitempty"`
	ModelType            string                  `json:"model_type,omitempty"`
	Capabilities         *gatewaypb.Capabilities `json:"capabilities,omitempty"`
	Properties           map[string]string       `json:"properties,omitempty"`
}

// catalogueFileSchema is the on-disk shape. An unparseable file is
// quarantined (renamed aside) and the catalogue starts empty; fields
// that no longer exist in catalogueEntry are silently dropped by the
// JSON decoder.
type catalogueFileSchema struct {
	Entries map[string]*catalogueEntry `json:"entries"`
}

// newMetadataCache constructs a catalogue rooted at dataDir, covering
// the model files under modelsDir. The disk file is loaded if present;
// missing or unreadable files start empty. A load that migrated legacy
// absolute keys is persisted immediately so the migration runs once.
func newMetadataCache(dataDir, modelsDir string, logger zerolog.Logger) *metadataCache {
	c := &metadataCache{
		logger:    logger.With().Str("component", "metadata-catalogue").Logger(),
		path:      filepath.Join(dataDir, catalogueFilename),
		modelsDir: modelsDir,
		entries:   map[string]*catalogueEntry{},
	}
	c.loadFromDisk()
	c.saveToDisk() // no-op unless load marked dirty (key migration)
	return c
}

// entryKey converts an absolute path under modelsDir into the
// slash-normalized modelsDir-relative key entries are stored under.
// ok=false when absPath does not live under modelsDir.
func (c *metadataCache) entryKey(absPath string) (string, bool) {
	rel, err := filepath.Rel(c.modelsDir, absPath)
	if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return "", false
	}
	return filepath.ToSlash(rel), true
}

// entryAbs resolves a catalogue key back to its absolute on-disk path.
func (c *metadataCache) entryAbs(key string) string {
	return filepath.Join(c.modelsDir, filepath.FromSlash(key))
}

// loadFromDisk populates entries from the catalogue file. Missing file
// is fine. Entries without a Name are dropped on load — a missing Name
// means the entry was never properly installed.
//
// One-time migration: catalogues written before keys went relative used
// absolute paths. Keys under the current modelsDir are converted to
// relative; entries outside it can't be expressed relative to the store
// and are dropped loudly.
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
		c.quarantineCorrupt(err)
		return
	}
	for k, v := range schema.Entries {
		if v == nil || strings.TrimSpace(v.Name) == "" {
			continue
		}
		key := k
		if filepath.IsAbs(k) || strings.HasPrefix(k, "/") {
			rel, ok := c.entryKey(filepath.Clean(filepath.FromSlash(k)))
			if !ok {
				c.logger.Warn().Str("path", k).Str("models_dir", c.modelsDir).Str("name", v.Name).
					Msg("dropping catalogue entry outside models dir during absolute→relative key migration")
				c.dirty = true
				continue
			}
			key = rel
			c.dirty = true
		}
		// Reservations persisted before ReservedAt existed carry a zero
		// timestamp; stamp them now so the TTL sweep can eventually
		// reap them instead of leaking forever.
		if v.Capabilities == nil && v.ReservedAt.IsZero() {
			v.ReservedAt = time.Now()
			c.dirty = true
		}
		c.entries[key] = v
	}
	c.logger.Debug().Int("entries", len(c.entries)).Msg("loaded model catalogue")
}

// quarantineCorrupt preserves an unparseable catalogue file instead of
// silently discarding it: operator-typed identity lives in there, and
// starting empty would otherwise be permanent loss. The bad file is
// renamed to <catalogue>.corrupt-<unixts> so the names can be recovered
// by hand; the gateway then starts with an empty catalogue.
func (c *metadataCache) quarantineCorrupt(parseErr error) {
	quarantine := fmt.Sprintf("%s.corrupt-%d", c.path, time.Now().Unix())
	if err := os.Rename(c.path, quarantine); err != nil {
		c.logger.Error().Err(parseErr).Str("path", c.path).AnErr("rename_err", err).
			Msg("catalogue file is corrupt and could not be quarantined; starting empty")
		return
	}
	c.logger.Error().Err(parseErr).Str("path", c.path).Str("quarantined_to", quarantine).
		Msg("catalogue file is corrupt; preserved aside and starting empty — operator-typed model names can be recovered from the quarantined file")
}

// saveToDisk persists the current map. Atomic write (temp + rename)
// so a crash mid-write doesn't leave a half-written file. No-op when
// clean.
//
// Marshalling happens under the lock: entries holds pointers that
// parseModelInfo / reserveEntry mutate in place, so serialising a
// shallow copy outside the lock would race those writes. The
// catalogue is tiny — only the file I/O runs unlocked.
func (c *metadataCache) saveToDisk() {
	c.mu.Lock()
	if !c.dirty {
		c.mu.Unlock()
		return
	}
	buf, err := json.MarshalIndent(catalogueFileSchema{Entries: c.entries}, "", "  ")
	if err != nil {
		c.mu.Unlock()
		c.logger.Warn().Err(err).Msg("marshalling catalogue; skipping save")
		return
	}
	c.dirty = false
	c.mu.Unlock()

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

// parameterCount returns the compute-relevant parameter count for the
// model at absPath — per-token active parameters for MoE models, the
// full count for dense ones — or 0 if the file isn't in the catalogue
// or the header didn't carry a usable count. predictCost callers use 0
// to fall back to a fallback parameter count.
//
// Lazily backfills missing counts: catalogues persisted from before the
// ActiveParameterCount field was added have entries with 0. We attempt
// one parseModelInfo call to re-read the GGUF header and populate the
// entry — the dispatch path will then see honest cost predictions on
// subsequent submissions without waiting for the operator to open the
// Models tab. When the re-read fails (file gone, header corrupt) the
// old total-count field is still better than nothing.
func (c *metadataCache) parameterCount(absPath string) uint64 {
	key, inStore := c.entryKey(absPath)
	if !inStore {
		return 0
	}
	c.mu.Lock()
	entry, ok := c.entries[key]
	c.mu.Unlock()
	if !ok {
		return 0
	}
	if entry.ActiveParameterCount > 0 {
		return entry.ActiveParameterCount
	}
	// Backfill: parseModelInfo re-reads the GGUF header into the entry
	// (it grabs the lock itself), and sets both parameter counts when
	// present. storeID is used only on the returned parsedModel which
	// the caller discards — basename is sufficient.
	if _, ok := c.parseModelInfo(absPath, filepath.Base(absPath)); ok {
		c.mu.Lock()
		defer c.mu.Unlock()
		if entry, ok := c.entries[key]; ok {
			if entry.ActiveParameterCount > 0 {
				return entry.ActiveParameterCount
			}
			return entry.ParameterCount
		}
		return 0
	}
	return entry.ParameterCount
}

// sha256For returns the cached lowercase-hex digest for the file at
// absPath, or "" when the file isn't in the catalogue or hashing failed.
// The gateway stamps it onto ModelFile.sha256 so a remote worker can
// verify integrity; "" makes the worker skip verification (safe degrade).
//
// Lazily backfills a missing hash: a load racing the first walk (or an
// entry from before this field existed) has an empty Sha256. One
// parseModelInfo call computes and stores it, mirroring parameterCount.
func (c *metadataCache) sha256For(absPath string) string {
	key, inStore := c.entryKey(absPath)
	if !inStore {
		return ""
	}
	c.mu.Lock()
	entry, ok := c.entries[key]
	if ok && entry.Sha256 != "" {
		sum := entry.Sha256
		c.mu.Unlock()
		return sum
	}
	c.mu.Unlock()
	if !ok {
		return ""
	}
	// Backfill: parseModelInfo hashes the file and stores it on the entry
	// (grabbing the lock itself). storeID is only used on the returned
	// parsedModel which we discard — basename suffices.
	if _, ok := c.parseModelInfo(absPath, filepath.Base(absPath)); !ok {
		return ""
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if entry, ok := c.entries[key]; ok {
		return entry.Sha256
	}
	return ""
}

// thinking reports the cached Capabilities.Thinking flag for absPath.
// Returns false when the entry is missing or the GGUF walk hasn't yet
// populated capabilities — that's the safe default for [predictCost]'s
// decode-token heuristic (plain chat-style estimate).
func (c *metadataCache) thinking(absPath string) bool {
	key, inStore := c.entryKey(absPath)
	if !inStore {
		return false
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	entry, ok := c.entries[key]
	if !ok || entry.Capabilities == nil {
		return false
	}
	return entry.Capabilities.GetThinking()
}

// properties returns a snapshot of the entry's GGUF-derived
// Properties map for absPath, or nil when the entry is missing.
// Used by the load-bytes estimator to read layers / embedding /
// head_count / head_count_kv.
func (c *metadataCache) properties(absPath string) map[string]string {
	key, inStore := c.entryKey(absPath)
	if !inStore {
		return nil
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	entry, ok := c.entries[key]
	if !ok || len(entry.Properties) == 0 {
		return nil
	}
	out := make(map[string]string, len(entry.Properties))
	for k, v := range entry.Properties {
		out[k] = v
	}
	return out
}

// companionMmprojPath returns the absolute path of the mmproj projector
// that shares primaryAbs's operator-typed Name, or "" when the primary
// has no Name yet or no sibling projector exists. The gateway uses this
// to auto-attach the vision projector at CHAT-load time so clients never
// have to know companion filenames — that's gateway-private knowledge
// about its own model store.
func (c *metadataCache) companionMmprojPath(primaryAbs string) string {
	primaryKey, inStore := c.entryKey(primaryAbs)
	if !inStore {
		return ""
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	primary, ok := c.entries[primaryKey]
	if !ok || primary.Name == "" || primary.Companion == "mmproj" {
		return ""
	}
	for key, other := range c.entries {
		if key == primaryKey || other == nil {
			continue
		}
		if other.Name == primary.Name && other.Companion == "mmproj" {
			return c.entryAbs(key)
		}
	}
	return ""
}

func (c *metadataCache) markDirty() {
	c.mu.Lock()
	c.dirty = true
	c.mu.Unlock()
}

// reserveEntry pre-writes the operator-supplied Name for one file.
// Called by planHFInstall / PlanLocalImport before bytes land on
// disk. Walk-time fields stay zero until the first walk fills them
// in; (mtime, size) staleness later triggers refresh.
//
// The reservation is persisted immediately: it is the only record tying
// an in-flight download to its operator-typed Name, and a crash before
// the next walk-driven save would otherwise orphan the file invisibly.
//
// Fails with errNameNotSluggable / errSlugCollision when name can't
// serve as a group identity (see assertSlugAvailable), and with
// errDestTaken when the destination is already occupied — a file on
// disk, or a live entry (reservation or committed) under a different
// Name. All enforced here, under the lock, so concurrent plans can't
// race a collision or a destination steal in. Re-reserving the same
// destination for the same Name is fine (a retried plan).
func (c *metadataCache) reserveEntry(absPath, name string) error {
	key, inStore := c.entryKey(absPath)
	if !inStore {
		return ctxerr.With(fmt.Errorf("reserve target outside models dir"), map[string]any{"path": absPath, "models_dir": c.modelsDir})
	}
	// Stat outside the lock (I/O): a file already at the destination
	// means it's taken no matter what the catalogue says.
	destOccupied := false
	if _, err := os.Stat(absPath); err == nil {
		destOccupied = true
	}
	if err := c.reserve(key, name, destOccupied); err != nil {
		return err
	}
	c.saveToDisk()
	return nil
}

// reserve is reserveEntry's under-lock core: validate, then write the
// reservation.
func (c *metadataCache) reserve(key, name string, destOccupied bool) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if err := c.assertSlugAvailable(name); err != nil {
		return err
	}
	if destOccupied {
		return fmt.Errorf("%w: %s", errDestTaken, key)
	}
	entry, ok := c.entries[key]
	if ok && entry.Name != name {
		return fmt.Errorf("%w: %s is claimed by group %q", errDestTaken, key, entry.Name)
	}
	if !ok {
		entry = &catalogueEntry{}
		c.entries[key] = entry
	}
	entry.Name = name
	if entry.Capabilities == nil {
		// Still a reservation — (re)stamp the clock so the TTL sweep
		// measures from the most recent plan.
		entry.ReservedAt = time.Now()
	}
	c.dirty = true
	return nil
}

// assertSlugAvailable rejects a group name that can't serve as a group
// identity: an empty slug (no ASCII letters/digits), or a slug already
// taken by a DIFFERENT existing Name. Re-using an existing Name
// verbatim is fine — that's how files join a group. exempt lists
// additional Names to ignore (renameGroup passes the outgoing name:
// its entries stop existing under that Name once the rename lands).
// Caller must hold c.mu.
func (c *metadataCache) assertSlugAvailable(name string, exempt ...string) error {
	slug := modelSlug(name)
	if slug == "" {
		return fmt.Errorf("%w: %q needs at least one letter or digit (a-z, 0-9)", errNameNotSluggable, name)
	}
	for _, e := range c.entries {
		if e == nil || e.Name == "" || e.Name == name {
			continue
		}
		if slices.Contains(exempt, e.Name) {
			continue
		}
		if modelSlug(e.Name) == slug {
			return fmt.Errorf("%w: %q and existing group %q would share the id %q — pick a distinct name", errSlugCollision, name, e.Name, slug)
		}
	}
	return nil
}

// renameGroup rewrites every catalogue entry whose current Name
// matches old to use newName. Returns the number of entries touched.
//
// Files stay where they are: the store-relative path is the instance
// key loaded workers hold (model.ID = "<relpath>#<hints-hash>"), so
// moving bytes on rename would strand resident instances and
// invalidate every id in flight. The slug directory a file lives in is
// an install-time artefact — published ids re-derive from the Name
// (parseModelInfo) and resolve back through storePathForID. Reserved
// entries (download in flight) rename freely: nothing moves, so there
// is nothing to orphan.
func (c *metadataCache) renameGroup(old, newName string) (int, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if old == "" || newName == "" || old == newName {
		return 0, nil
	}
	// The new name must be a usable, collision-free group identity for
	// the same reasons reserveEntry enforces it (Group.id = slug).
	// Entries still named old are exempt: they all move to newName, so
	// their current slug can't collide with the new one afterwards.
	if err := c.assertSlugAvailable(newName, old); err != nil {
		return 0, err
	}
	n := 0
	for _, e := range c.entries {
		if e == nil || e.Name != old {
			continue
		}
		e.Name = newName
		n++
	}
	if n > 0 {
		c.dirty = true
	}
	return n, nil
}

// storePathForID maps a published model id back to the physical
// under-format-root store path. Published ids carry the CURRENT group
// slug ("<modelSlug(Name)>/<filename>"), which stops matching the
// on-disk directory once the group has been renamed. Anything that
// doesn't resolve as such an id — a physical store path, an
// uncatalogued file, garbage — passes through unchanged.
func (c *metadataCache) storePathForID(p string) string {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.storePathForIDLocked(p)
}

// storePathForIDLocked is storePathForID's under-lock core, for
// callers already holding c.mu.
func (c *metadataCache) storePathForIDLocked(p string) string {
	slug, base := path.Dir(p), path.Base(p)
	if slug == "." || strings.Contains(slug, "/") {
		return p // published ids are exactly "<group-slug>/<filename>"
	}
	name := ""
	for _, e := range c.entries {
		if e != nil && e.Name != "" && modelSlug(e.Name) == slug {
			name = e.Name
			break
		}
	}
	if name == "" {
		return p
	}
	var keys []string
	for key, e := range c.entries {
		if e != nil && e.Name == name && path.Base(key) == base {
			keys = append(keys, key)
		}
	}
	if len(keys) == 0 {
		return p
	}
	// Prefer the key the id names verbatim (slug dir == physical dir);
	// otherwise pick deterministically.
	sort.Strings(keys)
	if phys := formatDir + "/" + p; slices.Contains(keys, phys) {
		return p
	}
	return strings.TrimPrefix(keys[0], formatDir+"/")
}

// groupRelPath returns the under-modelsDir relPath for srcName under
// groupName. A new group gets "<formatDir>/<group-slug>/<file>"; a
// group that already has entries keeps their directory even when its
// current name's slug differs (renames are catalogue-only — files stay
// put, and new files must land next to their siblings so a group never
// spans directories and filename collisions surface as errDestTaken).
func (c *metadataCache) groupRelPath(groupName, srcName string) string {
	base := sanitiseFilename(filepath.Base(srcName))
	c.mu.Lock()
	defer c.mu.Unlock()
	var keys []string
	for key, e := range c.entries {
		if e != nil && e.Name == groupName {
			keys = append(keys, key)
		}
	}
	if len(keys) == 0 {
		return formatDir + "/" + modelSlug(groupName) + "/" + base
	}
	sort.Strings(keys)
	return path.Dir(keys[0]) + "/" + base
}

// isPublishedID reports whether id has the shape of a published per-file
// id — "<group-slug>/<filename>", exactly one separator with non-empty
// parts. Distinguishes a target that's plausibly already deleted from a
// malformed one so PlanDelete can treat the former as an idempotent no-op.
func isPublishedID(id string) bool {
	slug, base := path.Dir(id), path.Base(id)
	return slug != "" && slug != "." && !strings.Contains(slug, "/") && base != ""
}

// relPathsForModel resolves a delete target — either a group slug
// (Group.id) or a single file's store id (Model.id, a relpath under
// formatRoot) — to the FULL store-relative catalogue keys of EVERY file
// sharing that model's Name (primary + companions). So deleting a vision
// chat model also removes its mmproj. Empty when nothing matches.
//
// Keys are returned verbatim — with the runtime-owned first segment
// (formatDir, "gguf/…") intact — because they ARE the paths MASS joins
// under its models root to remove the bytes and the same namespace as
// ModelFile.filename / DownloadFile.rel_path. Stripping the prefix here
// made RemoveLocal miss the file and silently no-op the delete. MASS
// removes the files; the gateway only decides.
func (c *metadataCache) relPathsForModel(id string) []string {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Resolve the target Name: match a group slug, or the entry for a
	// file whose store id (relpath under formatRoot) equals id. Published
	// per-file ids carry the current group slug rather than the physical
	// directory, so map them back first.
	targetName := ""
	idKey := formatDir + "/" + c.storePathForIDLocked(id)
	for key, e := range c.entries {
		if e == nil || e.Name == "" {
			continue
		}
		if modelSlug(e.Name) == id || key == idKey {
			targetName = e.Name
			break
		}
	}
	if targetName == "" {
		return nil
	}

	var out []string
	for key, e := range c.entries {
		if e == nil || e.Name != targetName {
			continue
		}
		out = append(out, key)
	}
	return out
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

// pruneAbsent drops every committed entry whose file is truly gone
// from disk. observed (keyed by modelsDir-relative key) lists the
// files the walk successfully parsed; an unobserved entry is only
// pruned when os.Stat confirms the file no longer exists — a file the
// walk failed to *read* (momentary AV-scanner lock, transient I/O
// error) keeps its entry and is retried on the next walk, because
// deleting it would permanently erase the operator-typed Name.
//
// Reserved entries — those without header data, pre-written by
// PlanLocalImport / PlanModelFiles before the file lands on disk —
// are kept so a download in flight (or an in-progress local copy)
// doesn't lose its reservation across walks, but only within
// reservationTTL: a reservation whose file never appeared is swept
// once it expires. "Committed" means Capabilities has been populated
// from a successful header read.
func (c *metadataCache) pruneAbsent(observed map[string]struct{}) {
	type candidate struct {
		key      string
		reserved bool
	}
	c.mu.Lock()
	var candidates []candidate
	for k, v := range c.entries {
		if _, ok := observed[k]; ok {
			continue
		}
		if v == nil {
			continue
		}
		if v.Capabilities == nil {
			// Reserved (no header read yet) — leave it for the file
			// to land, unless the reservation has expired.
			if v.ReservedAt.IsZero() || time.Since(v.ReservedAt) <= reservationTTL {
				continue
			}
			candidates = append(candidates, candidate{key: k, reserved: true})
			continue
		}
		candidates = append(candidates, candidate{key: k})
	}
	c.mu.Unlock()

	// Stat outside the lock (I/O), then delete under it.
	var gone []string
	for _, cand := range candidates {
		abs := c.entryAbs(cand.key)
		_, err := os.Stat(abs)
		switch {
		case err == nil && cand.reserved:
			// File landed but its header hasn't been read yet (e.g. the
			// walk keeps failing on it) — keep the reservation.
			c.logger.Warn().Str("path", abs).Msg("expired reservation's file exists but has no header data yet; keeping entry")
		case err == nil:
			c.logger.Warn().Str("path", abs).Msg("file exists but could not be parsed this walk; keeping catalogue entry and retrying next walk")
		case errors.Is(err, os.ErrNotExist):
			if cand.reserved {
				c.logger.Warn().Str("path", abs).Dur("ttl", reservationTTL).Msg("reservation expired without its file ever landing; dropping stale reservation")
			}
			gone = append(gone, cand.key)
		default:
			c.logger.Warn().Err(err).Str("path", abs).Msg("stat failed while pruning; keeping catalogue entry")
		}
	}
	if len(gone) == 0 {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	for _, k := range gone {
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
					if key, inStore := c.entryKey(info.AbsolutePath); inStore {
						observed[key] = struct{}{}
					}
					out = append(out, info)
				}
			}
			continue
		}
		if info, ok := c.maybeParseFile(root, d.Name()); ok {
			if key, inStore := c.entryKey(info.AbsolutePath); inStore {
				observed[key] = struct{}{}
			}
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
	// Skip MASS's in-progress download temp files.
	if strings.HasPrefix(name, downloadingPrefix) {
		return nil, false
	}
	return c.parseModelInfo(filepath.Join(dir, name), storeID)
}

// parseModelInfo returns the catalogue's view of one file. Files
// without a catalogue entry are logged and skipped — they weren't
// installed via PlanModelFiles / PlanLocalImport. Header is read on
// first observation or when (mtime, size) has drifted.
func (c *metadataCache) parseModelInfo(absPath, storeID string) (*parsedModel, bool) {
	key, inStore := c.entryKey(absPath)
	if !inStore {
		return nil, false
	}
	st, err := os.Stat(absPath)
	if err != nil {
		return nil, false
	}
	mtime, size := st.ModTime(), st.Size()

	c.mu.Lock()
	entry, hasEntry := c.entries[key]
	c.mu.Unlock()
	if !hasEntry {
		c.logger.Warn().Str("path", absPath).Str("hint", "register via Browse Local or HF install").Msg("file present without catalogue entry, ignoring")
		return nil, false
	}

	stale := !entry.MTime.Equal(mtime) || entry.Size != size
	// The parameter-count fields were added after the catalogue's first
	// ship (ActiveParameterCount later than ParameterCount) — old entries
	// had Capabilities populated but no counts, and the stale check above
	// wouldn't fire on them. Treat a missing count as cause to re-read
	// the header so the predictCost path stops falling back to
	// fallbackParameterCount (or a MoE total) on every dispatch.
	needsHeader := stale || entry.Capabilities == nil || entry.ParameterCount == 0 ||
		entry.ActiveParameterCount == 0 ||
		// The vision-encoder shape properties shipped after the catalogue
		// first learned about projectors: a populated mmproj entry from an
		// older binary has no vision_patch_size, so the image-token
		// estimate would silently keep the package defaults. A projector
		// genuinely lacking the key re-reads its (small) header once per
		// walk — cheap, and honest about what the file carries.
		(entry.Companion == "mmproj" && entry.Properties["vision_patch_size"] == "")
	// Sha256 backs remote-worker integrity checks and was added after the
	// catalogue's first ship. Recompute on staleness (a changed file must
	// get a fresh digest, not keep the old one) or when missing. Triggered
	// independently of needsHeader so a good header isn't re-parsed just to
	// backfill a hash — and a stale file recomputes both.
	needsHash := stale || entry.Sha256 == ""

	if needsHeader {
		meta, err := gguf.ReadMeta(absPath)
		if err != nil {
			// A reservation's file can be read mid-copy during a concurrent
			// install walk: the header is incomplete, this walk skips it, and
			// the entry stays a reservation (invisible in the list) until a
			// later walk re-reads the now-finished file. Warn, don't hide it at
			// Debug — a header that keeps failing is a real problem, not noise.
			c.logger.Warn().Err(err).Str("path", absPath).Msg("reading GGUF header; keeping entry, retrying next walk")
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
		entry.ParameterCount = meta.ParameterCount()
		entry.ActiveParameterCount = meta.ActiveParameterCount()
		entry.ReservedAt = time.Time{} // committed — no longer a reservation
		c.dirty = true
		c.mu.Unlock()
	}

	if needsHash {
		// Hash the full file OUTSIDE the lock — it's the one large I/O in
		// this path. On error, log and leave Sha256 empty: an empty hash
		// makes the worker skip verification (safe degrade), never fails
		// the walk.
		sum, err := hashFile(absPath)
		if err != nil {
			c.logger.Warn().Err(err).Str("path", absPath).Msg("hashing model file for integrity; leaving hash empty")
		} else {
			c.mu.Lock()
			entry.Sha256 = sum
			// mtime/size may not have been captured by the header branch
			// (hash-only backfill), so anchor the digest to the file state
			// it was computed from — otherwise a later stale check can't
			// tell whether the hash matches the current bytes.
			entry.MTime = mtime
			entry.Size = size
			c.dirty = true
			c.mu.Unlock()
		}
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
		for otherKey, other := range c.entries {
			if otherKey == key || other == nil {
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
	// The published id carries the CURRENT group slug, not the on-disk
	// directory: renames are catalogue-only (files never move), so after
	// one the directory's old slug would leak a stale group name through
	// every id. storePathForID maps published ids back to physical keys.
	id := storeID
	if entry.Name != "" {
		id = modelSlug(entry.Name) + "/" + path.Base(storeID)
	}
	return &parsedModel{
		ID:           id,
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

// hashFile returns the lowercase-hex SHA-256 of the full file at abs.
// Streams the file through the digest so a multi-GB GGUF never lands in
// memory. Any I/O error is returned so the caller can degrade to an
// empty hash (worker skips verification) without corrupting the entry.
func hashFile(abs string) (string, error) {
	f, err := os.Open(abs)
	if err != nil {
		return "", err
	}
	defer func() { _ = f.Close() }() // read-only: a Close error can't lose data
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
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
