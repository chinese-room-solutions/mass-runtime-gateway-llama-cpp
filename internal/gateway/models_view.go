package gateway

import (
	"fmt"
	"html"
	"path/filepath"
	"sort"
	"strings"

	gatewaypb "github.com/chinese-room-solutions/mass-proto/gen/go/gateway"
)

// groupModels collapses a flat list of parsed model files into the
// MASS-facing Group view. Identity is the operator-typed Name; same
// Name → same Group. Every catalogue entry under a Name becomes one
// child Model so the operator can see and individually manage every
// file (chat quants, projector companions, etc.). Group capability
// flags are the union across child Models — per-file truth lives on
// each child's own Capabilities. Returns groups sorted by display
// name for stable rendering.
func groupModels(infos []*parsedModel) []*gatewaypb.Group {
	type acc struct {
		types    []string // distinct non-companion model_types in insertion order
		seenType map[string]bool
		caps     gatewaypb.Capabilities
		models   []*gatewaypb.Model
	}
	groups := map[string]*acc{}
	var order []string

	for _, m := range infos {
		key := m.Name
		g, ok := groups[key]
		if !ok {
			g = &acc{seenType: map[string]bool{}}
			groups[key] = g
			order = append(order, key)
		}
		// Capability union across every file in the group.
		if m.Capabilities.GetVision() {
			g.caps.Vision = true
		}
		if m.Capabilities.GetAudio() {
			g.caps.Audio = true
		}
		if m.Capabilities.GetThinking() {
			g.caps.Thinking = true
		}
		// Collect distinct non-companion model_types (projectors
		// aren't a standalone type). First-seen order is preserved.
		if m.Companion == "" && m.ModelType != "" && !g.seenType[m.ModelType] {
			g.seenType[m.ModelType] = true
			g.types = append(g.types, m.ModelType)
		}
		g.models = append(g.models, &gatewaypb.Model{
			Id:           m.ID,
			Path:         m.AbsolutePath,
			SizeBytes:    m.SizeBytes,
			DisplayName:  modelDisplayName(filepath.Base(m.ID)),
			BadgeText:    m.VariantLabel,
			Capabilities: cloneCapabilities(m.Capabilities),
		})
	}

	out := make([]*gatewaypb.Group, 0, len(groups))
	for _, key := range order {
		g := groups[key]
		out = append(out, &gatewaypb.Group{
			Id:           modelSlug(key),
			DisplayName:  key,
			ModelTypes:   g.types,
			Capabilities: &gatewaypb.Capabilities{Vision: g.caps.Vision, Audio: g.caps.Audio, Thinking: g.caps.Thinking},
			Models:       g.models,
		})
	}
	sort.Slice(out, func(i, j int) bool {
		return strings.ToLower(out[i].GetDisplayName()) < strings.ToLower(out[j].GetDisplayName())
	})
	return out
}

// modelSlug converts the operator-typed Name into the stable opaque
// id MASS holds. The slug round-trips through nameForSlug to look the
// Name back up. Same Name → same slug.
func modelSlug(name string) string {
	var b strings.Builder
	b.Grow(len(name))
	prevHyphen := false
	for _, r := range strings.TrimSpace(strings.ToLower(name)) {
		switch {
		case r >= 'a' && r <= 'z', r >= '0' && r <= '9':
			b.WriteRune(r)
			prevHyphen = false
		default:
			if !prevHyphen && b.Len() > 0 {
				b.WriteByte('-')
				prevHyphen = true
			}
		}
	}
	return strings.Trim(b.String(), "-")
}

// modelDisplayName is the prettified label MASS shows on the model
// row. We render the source-faithful filename without its extension
// — preserves casing the operator/source chose.
func modelDisplayName(filename string) string {
	return strings.TrimSuffix(filename, filepath.Ext(filename))
}

// renderModelDetail returns the right-pane properties panel HTML for
// one model file. Reads catalogue-stored identity verbatim — no
// filename re-parsing.
//
// Returns "" when the model isn't found — the caller renders a tiny
// "model not found" panel rather than leaving the area blank.
func renderModelDetail(runtimeName string, info *parsedModel) string {
	esc := html.EscapeString
	filename := filepath.Base(filepath.FromSlash(info.ID))
	modelDisplay := modelDisplayName(filename)
	groupName := info.Name
	if groupName == "" {
		groupName = modelDisplay
	}
	modelType := titleCaseType(info.ModelType)

	var b strings.Builder
	b.WriteString(`<div class="bg-neutral-800/60 rounded-lg border border-neutral-700/50 p-5" style="display:flex;flex-direction:column">`)

	// Header (Group name + Model filename).
	b.WriteString(`<div class="flex items-start justify-between mb-4"><div class="min-w-0 flex-1">`)
	fmt.Fprintf(&b, `<h3 class="text-sm font-semibold text-white">%s</h3>`, esc(groupName))
	fmt.Fprintf(&b, `<p class="text-xs text-neutral-400 mt-0.5">%s</p></div>`, esc(modelDisplay))
	b.WriteString(`<sl-icon-button name="x-lg" style="font-size:0.85rem" data-on:click="$selectedModelID=''"></sl-icon-button>`)
	b.WriteString(`</div>`)

	// Identity.
	b.WriteString(`<div class="space-y-2.5">`)
	writeDetailProp(&b, "Group", groupName)
	writeDetailProp(&b, "Model", modelDisplay)
	if modelType != "" {
		writeDetailProp(&b, "Type", modelType)
	}
	if info.VariantLabel != "" {
		writeDetailProp(&b, "Quantization", info.VariantLabel)
	}
	writeDetailProp(&b, "Format", "GGUF")
	writeDetailProp(&b, "Size", humanSize(info.SizeBytes))
	writeDetailProp(&b, "Runtime", prettyRuntimeName(runtimeName))
	b.WriteString(`</div>`)

	// Custom properties (architecture, layers, context, etc.).
	if len(info.Properties) > 0 {
		keys := make([]string, 0, len(info.Properties))
		for k := range info.Properties {
			// "architecture" / "name" surface as first-class fields elsewhere
			// (or are uninteresting duplicates of the filename).
			if k == "architecture" || k == "name" {
				continue
			}
			keys = append(keys, k)
		}
		sort.Strings(keys)
		if len(keys) > 0 {
			b.WriteString(`<div class="border-t border-neutral-700/50 my-3"></div><div class="space-y-2.5">`)
			labels := map[string]string{
				"context":   "Context Length",
				"embedding": "Embedding Dim",
				"layers":    "Layers",
				"vocab":     "Vocab Size",
				"tensors":   "Tensors",
			}
			for _, k := range keys {
				label := labels[k]
				if label == "" {
					label = titleCaseWords(strings.ReplaceAll(k, "_", " "))
				}
				writeDetailProp(&b, label, info.Properties[k])
			}
			b.WriteString(`</div>`)
		}
	}

	// File path / model id.
	b.WriteString(`<div class="pt-3 mt-3 border-t border-neutral-700/50">`)
	b.WriteString(`<div class="flex items-center gap-1 mb-1"><span class="text-xs text-neutral-400">Model ID</span></div>`)
	fmt.Fprintf(&b, `<code class="text-xs text-neutral-300 font-mono break-all leading-relaxed">%s</code>`, esc(info.ID))
	b.WriteString(`</div>`)

	// Delete action — same signal flow as the row's trash icon.
	b.WriteString(`<div class="pt-4 border-t border-neutral-700/50 flex items-center justify-center gap-3" style="margin-top:0.75rem">`)
	fmt.Fprintf(&b, `<sl-button size="small" variant="danger" outline data-on:click="$confirmDeleteModelID='%s'; $confirmDeleteModelKind='%s'; $confirmDeleteModelOpen=true"><sl-icon slot="prefix" name="trash3"></sl-icon>Delete</sl-button>`,
		jsStringEscape(info.ID), jsStringEscape(runtimeName))
	b.WriteString(`</div>`)

	b.WriteString(`</div>`)
	return b.String()
}

func writeDetailProp(b *strings.Builder, label, value string) {
	esc := html.EscapeString
	fmt.Fprintf(b, `<div class="flex items-baseline gap-3"><span class="text-xs text-neutral-400 flex-shrink-0 whitespace-nowrap" style="min-width:8.5rem">%s</span><span class="text-xs text-neutral-200 font-mono break-all min-w-0">%s</span></div>`,
		esc(label), esc(value))
}

// titleCaseWords capitalizes the first rune of each space-separated word.
func titleCaseWords(s string) string {
	words := strings.Fields(s)
	for i, w := range words {
		if w == "" {
			continue
		}
		words[i] = strings.ToUpper(w[:1]) + w[1:]
	}
	return strings.Join(words, " ")
}

// titleCaseType normalises gateway-supplied "chat" / "embedding" labels
// for display.
func titleCaseType(t string) string {
	switch strings.ToLower(t) {
	case "chat":
		return "Chat"
	case "embedding":
		return "Embedding"
	case "mmproj":
		return "Mmproj"
	default:
		if t == "" {
			return ""
		}
		return strings.ToUpper(t[:1]) + t[1:]
	}
}

// prettyRuntimeName turns "llama-cpp" into "llama.cpp" for the pill.
// Other kinds fall back verbatim.
func prettyRuntimeName(kind string) string {
	if kind == "llama-cpp" {
		return "llama.cpp"
	}
	return kind
}

func humanSize(bytes int64) string {
	const (
		KB = 1024
		MB = KB * 1024
		GB = MB * 1024
	)
	switch {
	case bytes >= GB:
		return fmt.Sprintf("%.1f GB", float64(bytes)/float64(GB))
	case bytes >= MB:
		return fmt.Sprintf("%.1f MB", float64(bytes)/float64(MB))
	case bytes >= KB:
		return fmt.Sprintf("%.1f KB", float64(bytes)/float64(KB))
	default:
		return fmt.Sprintf("%d B", bytes)
	}
}

func jsStringEscape(s string) string {
	r := strings.NewReplacer(`\`, `\\`, `'`, `\'`, "\n", `\n`, "\r", `\r`)
	return r.Replace(s)
}
