package gateway

import (
	"encoding/json"
	"fmt"
	"html"
	"net/http"
	"net/url"
	"strings"
	"sync"

	"github.com/chinese-room-solutions/mass-sdk/hfui"
	hf "github.com/chinese-room-solutions/mass-sdk/huggingface"
	"github.com/chinese-room-solutions/mass-sdk/uikit"
)

// installPageSize matches MASS's pre-shift "Show More" page size.
const installPageSize = 5

// installSearchState holds per-query pagination state for the
// install UI's Show More flow. Single-process state is fine — the
// gateway is the only thing serving these.
type installSearchState struct {
	cursor   string
	shownIDs []string
	hasMore  bool
}

// installSearchStateCap bounds how many per-query pagination states
// the gateway keeps. Every distinct search creates one; without a cap
// the map grows for the lifetime of the process. When full, the
// oldest query is evicted (its Show More button then just renders an
// empty footer — the operator re-searches).
const installSearchStateCap = 100

var (
	installSearchMu    sync.Mutex
	installSearchByKey = map[string]*installSearchState{}
	installSearchOrder []string // insertion order of keys, oldest first
)

// storeInstallSearchState remembers state for query q, evicting the
// oldest queries past installSearchStateCap.
func storeInstallSearchState(q string, s *installSearchState) {
	installSearchMu.Lock()
	defer installSearchMu.Unlock()
	if _, exists := installSearchByKey[q]; !exists {
		installSearchOrder = append(installSearchOrder, q)
		for len(installSearchOrder) > installSearchStateCap {
			delete(installSearchByKey, installSearchOrder[0])
			installSearchOrder = installSearchOrder[1:]
		}
	}
	installSearchByKey[q] = s
}

// installRouter mounts the gateway-hosted install UI under /.install*.
// MASS's iframe modal points at /mass.<runtime_name>.*/install (the
// dot prefix survives MASS's path strip, then gatewayhttp prepends
// the leading slash). The page hosts an HF search box; on Get-click
// it prompts for the operator-typed group name and POSTs the
// resolved file list back to MASS via MassScheduler.DownloadFiles.
func installRouter(d routerDeps) http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /.install", func(w http.ResponseWriter, r *http.Request) {
		theme := uikit.ThemeFromRequest(r)
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.Header().Set("Cache-Control", "no-store, no-cache, must-revalidate")
		// The page loads from MASS's origin, so its asset URLs must route
		// back through the /mass.<runtime> proxy to this gateway's
		// AssetsHandler (mounted in newRouter).
		_, _ = w.Write([]byte(uikit.LayoutUnder("/mass."+d.runtimeName, "Install Model", installPageBody(d.runtimeName), theme)))
	})
	mux.HandleFunc("POST /.install/search", func(w http.ResponseWriter, r *http.Request) {
		handleInstallSearch(w, r, d)
	})
	mux.HandleFunc("POST /.install/more", func(w http.ResponseWriter, r *http.Request) {
		handleInstallMore(w, r, d)
	})
	mux.HandleFunc("POST /.install/submit", func(w http.ResponseWriter, r *http.Request) {
		handleInstallSubmit(w, r, d)
	})
	return mux
}

// installPageBody is the HTML body of the install page. Search
// row sits at the top; results scroll inside #install-results.
// MASS gives the iframe a fixed-height container — the gateway
// owns layout and scroll behaviour entirely. The only signal the
// page sends back is `mass-install-queued` after a successful
// submit, which closes the MASS dialog.
func installPageBody(runtimeName string) string {
	return `
<style>
	html, body, body.sl-theme-dark {
		background: var(--mass-bg-base) !important;
		margin: 0 !important;
		padding: 0 !important;
	}
	/* Pane layout: search row pinned at the top, results scroll
	   internally once the page hits its self-imposed row cap, footer
	   (Show More) trails the rows. The page reports its desired
	   height back to MASS — empty state collapses to just the search
	   row, populated state grows up to the cap. */
	#install-root {
		display: flex;
		flex-direction: column;
		padding: 0 0.75rem;
		box-sizing: border-box;
	}
	#install-results {
		flex: 1 1 auto;
		min-height: 0;
		overflow-y: auto;
		margin-top: 0.5rem;
	}
	#install-footer {
		flex: 0 0 auto;
		padding: 0.5rem 0;
	}
	/* The in-iframe name-prompt sl-dialog overlays everything
	   when shown; pin it out of document flow so it doesn't add
	   to the page's reported height when hidden. */
	#install-name-dialog {
		position: absolute;
		top: 0;
		left: 0;
	}
</style>
<div id="install-root">
	<div class="flex gap-2" style="flex:0 0 auto">
		<sl-input id="install-query" size="small" clearable placeholder="Search Hugging Face models..." class="flex-1" autocomplete="off"></sl-input>
		<sl-button variant="primary" size="small" id="install-search-btn">
			<sl-icon slot="prefix" name="search"></sl-icon>
			Search
		</sl-button>
	</div>
	<div id="install-results"></div>
	<div id="install-footer"></div>
</div>

<sl-dialog id="install-name-dialog" class="mass-dialog" label="Name this group" style="--width:420px">
	<div class="mb-3 text-xs" style="color:var(--mass-text-muted)" id="install-name-subtitle"></div>
	<sl-input id="install-name-input" size="small" placeholder="e.g. Qwen3 30B" autocomplete="off"></sl-input>
	<sl-button slot="footer" size="small" id="install-name-cancel">Cancel</sl-button>
	<sl-button slot="footer" size="small" variant="primary" id="install-name-ok" disabled>Install</sl-button>
</sl-dialog>

<script>
(function() {
	var runtimeName = ` + jsonString(runtimeName) + `;
	// Page URL doubles as the route prefix: MASS strips /mass.<runtime>
	// from incoming requests, so /mass.<runtime>.install/search maps to
	// the gateway's /.install/search route. Reading our own pathname
	// keeps the gateway from hardcoding MASS's URL conventions.
	var prefix = window.location.pathname;
	var queryEl = document.getElementById('install-query');
	var resultsEl = document.getElementById('install-results');
	var footerEl = document.getElementById('install-footer');
	var btn = document.getElementById('install-search-btn');
	var nameDlg = document.getElementById('install-name-dialog');
	var nameInput = document.getElementById('install-name-input');
	var nameOK = document.getElementById('install-name-ok');
	var nameCancel = document.getElementById('install-name-cancel');
	var nameSubtitle = document.getElementById('install-name-subtitle');

	// The page tells MASS how tall it wants its iframe to be. Empty
	// state collapses to just the search row; populated state grows
	// with rows up to a cap, after which #install-results scrolls
	// internally. MASS only enforces a viewport ceiling — the cap
	// itself is this page's choice.
	var visibleRowCap = 10;
	var rootEl = document.getElementById('install-root');
	// MASS tags each load with a session id (?s=N). We echo it back
	// so MASS can drop resize messages from prior sessions (the old
	// page can still post a final size as it's torn down).
	var session = new URLSearchParams(window.location.search).get('s');

	function reportHeight() {
		if (window.parent === window || !rootEl) return;
		// Cap the results pane at N rows so its overflow:auto kicks
		// in past that. We measure live rows (heights vary with name
		// length and badge count) instead of guessing a constant.
		var rows = resultsEl.querySelectorAll('[data-hf-row]');
		if (rows.length > visibleRowCap) {
			var firstTop = rows[0].getBoundingClientRect().top;
			var nthBottom = rows[visibleRowCap - 1].getBoundingClientRect().bottom;
			resultsEl.style.maxHeight = Math.ceil(nthBottom - firstTop) + 'px';
		} else {
			resultsEl.style.maxHeight = '';
		}
		// Use the wrapper's actual content size, not
		// documentElement.scrollHeight: scrollHeight is bounded
		// below by the iframe element's own height, so feeding it
		// back to MASS makes the iframe size monotonically grow and
		// never shrink. The wrapper rect tracks the real content.
		var h = Math.ceil(rootEl.getBoundingClientRect().height);
		window.parent.postMessage({type: 'mass-install-resize', height: h, session: session}, '*');
	}
	if (window.ResizeObserver && rootEl) {
		new ResizeObserver(reportHeight).observe(rootEl);
	}
	window.addEventListener('load', reportHeight);

	// Initial search returns HTML for both rows and footer, separated
	// by a sentinel comment. The Show More button (rendered server-
	// side) carries Datastar's @post(URL) directive; subsequent pages
	// stream back as Datastar patches without any custom JS here.
	function search() {
		// Clear the row-cap clamp from the previous search so the
		// pane can naturally collapse to fit fewer results — without
		// this, the new (smaller) result set still renders inside a
		// container locked to the previous tall max-height and the
		// ResizeObserver never sees a size change to fire on.
		resultsEl.style.maxHeight = '';
		var q = (queryEl.value || '').trim();
		if (!q) { resultsEl.innerHTML = ''; footerEl.innerHTML = ''; return; }
		resultsEl.innerHTML = '<div class="flex items-center justify-center py-6"><sl-spinner style="font-size:1.5rem"></sl-spinner></div>';
		footerEl.innerHTML = '';
		var fd = new FormData();
		fd.append('query', q);
		fetch(prefix + '/search', {method: 'POST', body: fd})
			.then(function(r) { return r.text(); })
			.then(function(html) {
				var parts = html.split('<!--mass-install-split-->');
				resultsEl.innerHTML = parts[0] || '';
				footerEl.innerHTML = parts[1] || '';
				// innerHTML never executes <script> elements, and the hfui
				// rows ship their runtime (__hfOpen, __hfDlID, ...) inline.
				// Re-create each script node so the runtime actually runs.
				resultsEl.querySelectorAll('script').forEach(function(old) {
					var s = document.createElement('script');
					for (var i = 0; i < old.attributes.length; i++) {
						s.setAttribute(old.attributes[i].name, old.attributes[i].value);
					}
					s.textContent = old.textContent;
					old.parentNode.replaceChild(s, old);
				});
				// The hfui runtime just clobbered __hfDownload with its
				// default (a direct POST to a MASS endpoint that doesn't
				// exist here) — re-assert the name-prompt flow.
				window.__hfDownload = promptDownload;
				// Force a height report after the result-set swap.
				// ResizeObserver only fires when the observed element
				// actually changes size; replacing N rows with M rows
				// can leave #install-root the same dimensions if N≈M
				// or if the previous max-height clamp was hiding the
				// difference. Direct call closes that gap.
				reportHeight();
			})
			.catch(function(err) {
				resultsEl.innerHTML = '<sl-alert variant="danger" open>Search failed: ' + String(err).replace(/</g,'&lt;') + '</sl-alert>';
			});
	}
	btn.onclick = search;
	queryEl.addEventListener('keydown', function(e) {
		if (e.key === 'Enter') { e.preventDefault(); search(); }
	});

	// Name prompt mirrors the pre-shift MASS flow: clicking Get on a
	// file pops a prompt asking for the group name, then ships the
	// install to MASS via {prefix}/submit (which calls DownloadFiles).
	var pending = null; // {repoID, filename}
	function refreshOK() { nameOK.disabled = nameInput.value.trim() === ''; }
	nameInput.addEventListener('input', refreshOK);
	nameInput.addEventListener('keydown', function(e) {
		if (e.key === 'Enter' && !nameOK.disabled) { e.preventDefault(); nameOK.click(); }
	});
	nameCancel.onclick = function() {
		if (pending && window.__hfDlCancel) window.__hfDlCancel(pending.filename);
		pending = null;
		nameDlg.hide();
	};
	nameOK.onclick = function() {
		var groupName = nameInput.value.trim();
		if (!groupName || !pending) return;
		var p = pending;
		pending = null;
		nameDlg.hide();
		var id = window.__hfDlID && window.__hfDlID(p.filename);
		if (id) {
			document.querySelectorAll('[id="' + id + '"]').forEach(function(el) {
				el.innerHTML = '<sl-spinner style="font-size:1rem"></sl-spinner>';
			});
		}
		fetch(prefix + '/submit', {
			method: 'POST',
			headers: {'Content-Type': 'application/json'},
			body: JSON.stringify({repo_id: p.repoID, filename: p.filename, group_name: groupName})
		}).then(function(r) {
			if (!r.ok) return r.text().then(function(t) { throw new Error(t || ('HTTP ' + r.status)); });
			return r.json();
		}).then(function() {
			if (window.__hfDlDone) window.__hfDlDone(p.filename);
			if (window.parent !== window) {
				window.parent.postMessage({type: 'mass-install-queued', runtime: runtimeName}, '*');
			}
		}).catch(function(err) {
			if (window.__hfDlErr) window.__hfDlErr(p.filename, err.message);
		});
	};

	// hfui's rows wire Get buttons to window.__hfDownload. Open the
	// name prompt instead of POSTing immediately. Re-assigned after
	// every search injection because the hfui runtime script defines
	// its own default over it.
	function promptDownload(repoID, filename) {
		pending = {repoID: repoID, filename: filename};
		// The variant picker overlay sits at z-index 9999, above
		// sl-dialog's default 800, so it would cover the prompt. Its
		// job is done once a variant is picked — close it. Progress /
		// done / error state still lands in the row's <template> copy,
		// so reopening the picker shows the current state.
		if (window.__hfClose) window.__hfClose();
		nameSubtitle.textContent = filename;
		nameInput.value = '';
		refreshOK();
		nameDlg.show();
		setTimeout(function() { nameInput.focus(); }, 50);
	}
	window.__hfDownload = promptDownload;

})();
</script>`
}

// installSplit is the sentinel the search response uses to divide
// results-rows HTML from footer HTML in a single response body.
const installSplit = "<!--mass-install-split-->"

// handleInstallSearch runs a fresh HF search and returns rendered
// rows + a Datastar-bound Show More footer in one response. The
// page splits on installSplit and injects each half into its own
// container.
func handleInstallSearch(w http.ResponseWriter, r *http.Request, d routerDeps) {
	q := strings.TrimSpace(r.FormValue("query"))
	if q == "" {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		_, _ = w.Write([]byte(installSplit))
		return
	}
	res, err := hf.Search(r.Context(), q, hf.SearchOptions{
		Limit:    installPageSize,
		FileExts: []string{".gguf"},
	})
	if err != nil {
		d.logger.Warn().Err(err).Str("query", q).Msg("HF search failed")
		writeInstallSearch(w, `<sl-alert variant="danger" open>Search failed: `+html.EscapeString(err.Error())+`</sl-alert>`, "")
		return
	}
	state := &installSearchState{cursor: res.NextCursor, hasMore: res.HasMore}
	for _, m := range res.Models {
		state.shownIDs = append(state.shownIDs, m.RepoID)
	}
	storeInstallSearchState(q, state)
	rows := hfui.RenderResults(d.runtimeName, toUIKitModels(res.Models), hfui.ResultsOpts{
		SkipFooter: true,
	})
	footer := installFooterHTML(d.runtimeName, q, res.HasMore)
	writeInstallSearch(w, rows, footer)
}

// handleInstallMore streams the next page back as Datastar patches:
// append rows into uikit's #pe-hf-list and outer-replace
// #install-footer with the new Show More button (or empty when no
// more pages remain).
func handleInstallMore(w http.ResponseWriter, r *http.Request, d routerDeps) {
	q := strings.TrimSpace(r.FormValue("query"))
	installSearchMu.Lock()
	state := installSearchByKey[q]
	installSearchMu.Unlock()

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	flusher, _ := w.(http.Flusher)
	emit := func(selector, mode, body string) {
		_, _ = fmt.Fprintf(w,
			"event: datastar-patch-elements\ndata: selector %s\ndata: mode %s\ndata: elements %s\n\n",
			selector, mode, body)
		if flusher != nil {
			flusher.Flush()
		}
	}

	if state == nil || !state.hasMore {
		emit("#install-footer", "outer", `<div id="install-footer"></div>`)
		return
	}
	res, err := hf.Search(r.Context(), q, hf.SearchOptions{
		Limit:      installPageSize,
		FileExts:   []string{".gguf"},
		Cursor:     state.cursor,
		ExcludeIDs: state.shownIDs,
	})
	if err != nil {
		d.logger.Warn().Err(err).Str("query", q).Msg("HF more failed")
		emit("#install-footer", "outer", `<div id="install-footer"><sl-alert variant="danger" open>Search failed: `+html.EscapeString(err.Error())+`</sl-alert></div>`)
		return
	}
	for _, m := range res.Models {
		state.shownIDs = append(state.shownIDs, m.RepoID)
	}
	state.cursor = res.NextCursor
	state.hasMore = res.HasMore
	storeInstallSearchState(q, state)

	rows := hfui.RenderResultRows(toUIKitModels(res.Models), nil)
	if rows != "" {
		emit("#pe-hf-list", "append", rows)
	}
	emit("#install-footer", "outer", installFooterHTML(d.runtimeName, q, res.HasMore))
}

// installFooterHTML renders the Show More footer. The button binds
// to Datastar's @post directive so clicking streams the next page
// back as Datastar patches — no custom JS in the page handles this.
// Empty footer when there are no more results.
//
// The URL must be MASS-facing, not the gateway-internal post-strip
// path: Datastar fires the POST from the iframe browser context,
// which sends `/mass.<runtime>.install/more` to MASS, which strips
// the prefix and forwards `.install/more` back to us.
func installFooterHTML(runtimeName, query string, hasMore bool) string {
	if !hasMore {
		return `<div id="install-footer"></div>`
	}
	moreURL := "/mass." + runtimeName + ".install/more?query=" + url.QueryEscape(query)
	// data-indicator flips $installLoading while the @post is in
	// flight; we hide the button and show a spinner via data-show.
	// Once the SSE patch lands (replacing this whole footer), the
	// signal value is irrelevant for the next render.
	return `<div id="install-footer" class="text-center py-2" data-signals="{installLoading:false}">` +
		`<sl-button size="small" variant="text" data-show="!$installLoading" ` +
		`data-on:click="@post('` + html.EscapeString(moreURL) + `')" data-indicator="installLoading">` +
		`<sl-icon slot="prefix" name="chevron-down"></sl-icon>Show More</sl-button>` +
		`<sl-spinner data-show="$installLoading" style="font-size:1.2rem"></sl-spinner>` +
		`</div>`
}

// writeInstallSearch writes the search response body: rows HTML +
// sentinel + footer HTML so the page can split and inject each
// half into its own container.
func writeInstallSearch(w http.ResponseWriter, rows, footer string) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	_, _ = w.Write([]byte(rows + installSplit + footer))
}

// toUIKitModels projects mass-sdk huggingface.Model into the SDK's
// hfui.ResultModel render shape.
func toUIKitModels(in []hf.Model) []hfui.ResultModel {
	out := make([]hfui.ResultModel, 0, len(in))
	for _, m := range in {
		files := make([]hfui.ResultFile, 0, len(m.Files))
		for _, f := range m.Files {
			files = append(files, hfui.ResultFile{Filename: f.Filename, SizeBytes: f.SizeBytes})
		}
		out = append(out, hfui.ResultModel{
			RepoID:      m.RepoID,
			Description: m.Description,
			Downloads:   m.Downloads,
			Likes:       m.Likes,
			Params:      m.Params,
			PipelineTag: m.PipelineTag,
			AvatarURL:   m.AvatarURL,
			Files:       files,
		})
	}
	return out
}

// handleInstallSubmit is the bridge: take the operator's pick, plan
// the file set (primary + mmproj companion), and ship the list to
// MASS's downloads manager via MassScheduler.DownloadFiles.
func handleInstallSubmit(w http.ResponseWriter, r *http.Request, d routerDeps) {
	var req struct {
		RepoID    string `json:"repo_id"`
		Filename  string `json:"filename"`
		GroupName string `json:"group_name"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "bad body: "+err.Error(), http.StatusBadRequest)
		return
	}
	files, err := planHFInstall(r.Context(), d.modelsDir, d.cache, req.RepoID, req.Filename, req.GroupName)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	queued, err := d.scheduler.DownloadFiles(r.Context(), req.GroupName, files)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{"queued": queued})
}

// jsonString quotes s as a JSON string literal so it can be inlined
// in the page's <script> block without escaping concerns.
func jsonString(s string) string {
	b, err := json.Marshal(s)
	if err != nil {
		return `""`
	}
	return string(b)
}
