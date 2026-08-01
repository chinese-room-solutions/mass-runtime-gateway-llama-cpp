package gateway

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/require"
)

// The install UI's per-query pagination state must stay bounded: every
// distinct search adds a map entry, so an operator (or a scripted
// caller) hammering searches would otherwise grow the map for the
// process lifetime. Past installSearchStateCap the oldest query is
// evicted; updates to an existing query never evict.
func TestStoreInstallSearchState_BoundedEviction(t *testing.T) {
	// Package-level state: isolate and restore.
	savedByKey, savedOrder := installSearchByKey, installSearchOrder
	t.Cleanup(func() {
		installSearchByKey, installSearchOrder = savedByKey, savedOrder
	})
	installSearchByKey = map[string]*installSearchState{}
	installSearchOrder = nil

	total := installSearchStateCap + 50
	for i := range total {
		storeInstallSearchState(fmt.Sprintf("query-%d", i), &installSearchState{cursor: fmt.Sprint(i)})
	}

	require.Len(t, installSearchByKey, installSearchStateCap)
	require.Len(t, installSearchOrder, installSearchStateCap)
	require.NotContains(t, installSearchByKey, "query-0", "oldest queries are evicted")
	require.NotContains(t, installSearchByKey, fmt.Sprintf("query-%d", total-installSearchStateCap-1))
	require.Contains(t, installSearchByKey, fmt.Sprintf("query-%d", total-installSearchStateCap), "newest cap-many queries survive")
	require.Contains(t, installSearchByKey, fmt.Sprintf("query-%d", total-1))

	// Updating an existing key replaces state without growing or evicting.
	survivor := fmt.Sprintf("query-%d", total-installSearchStateCap)
	storeInstallSearchState(fmt.Sprintf("query-%d", total-1), &installSearchState{cursor: "updated"})
	require.Len(t, installSearchByKey, installSearchStateCap)
	require.Contains(t, installSearchByKey, survivor, "update must not evict")
	require.Equal(t, "updated", installSearchByKey[fmt.Sprintf("query-%d", total-1)].cursor)
}
