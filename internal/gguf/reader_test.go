package gguf

import (
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

// parseSizeLabel handles GGUF general.size_label strings — short tags
// like "8B" or "1.5B" published by the model card. Falls back to 0 on
// unrecognised input so callers can choose a different default. Bad
// fractional inputs are explicitly enumerated since the gateway falls
// back to fallbackParameterCount on a 0 return — a silent miscount
// would mislabel its size and skew scoring.
func TestParseSizeLabel(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want uint64
	}{
		{name: "8B", in: "8B", want: 8_000_000_000},
		{name: "lower-case b", in: "70b", want: 70_000_000_000},
		{name: "fractional 1.5B", in: "1.5B", want: 1_500_000_000},
		{name: "millions", in: "770M", want: 770_000_000},
		{name: "thousands", in: "10K", want: 10_000},
		{name: "trillions", in: "1T", want: 1_000_000_000_000},
		{name: "empty", in: "", want: 0},
		{name: "no suffix", in: "8", want: 0},
		{name: "garbage", in: "huge", want: 0},
		{name: "two-digit fractional", in: "13.25B", want: 13_250_000_000},
		// Bad fractional whole part — non-digit before the dot.
		{name: "non-numeric whole part", in: "x.5B", want: 0},
		// Bad fractional after the dot — non-digit, not just empty.
		{name: "non-numeric fractional part", in: "1.xB", want: 0},
		// Empty whole part (".5B") — parseUintLoose returns
		// non-digit. Falls back to 0 even though it could be 500M.
		// Documents that we treat malformed size labels conservatively.
		{name: "leading dot", in: ".5B", want: 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			require.Equal(t, tt.want, parseSizeLabel(tt.in))
		})
	}
}

// Meta.ParameterCount prefers general.parameter_count over
// general.size_label; size_label is one-sig-fig (vendor-published) but
// parameter_count is exact. Falls back to size_label parsing when
// parameter_count is absent or zero. Returns 0 only when neither key
// is present — gateway then uses its fallbackParameterCount.
func TestMetaParameterCount(t *testing.T) {
	tests := []struct {
		name string
		kv   map[string]any
		want uint64
	}{
		{
			// Exact parameter_count present — used verbatim.
			name: "uint64 parameter_count wins",
			kv:   map[string]any{"general.parameter_count": uint64(8_030_261_248)},
			want: 8_030_261_248,
		},
		{
			// uint32 also accepted via GetUint64's switch.
			name: "uint32 parameter_count widens",
			kv:   map[string]any{"general.parameter_count": uint32(770_000_000)},
			want: 770_000_000,
		},
		{
			// parameter_count zero (some older GGUFs emit it as 0):
			// fall through to size_label.
			name: "zero parameter_count falls through",
			kv:   map[string]any{"general.parameter_count": uint64(0), "general.size_label": "8B"},
			want: 8_000_000_000,
		},
		{
			// No parameter_count — size_label alone.
			name: "size_label only",
			kv:   map[string]any{"general.size_label": "1.5B"},
			want: 1_500_000_000,
		},
		{
			// Neither key — caller falls back to a median size.
			name: "both absent → 0",
			kv:   map[string]any{},
			want: 0,
		},
		{
			// Garbage size_label with no parameter_count still
			// returns 0; predictCost uses fallbackParameterCount.
			name: "garbage size_label → 0",
			kv:   map[string]any{"general.size_label": "Large"},
			want: 0,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := &Meta{KV: tt.kv}
			require.Equal(t, tt.want, m.ParameterCount())
		})
	}
}

// ParameterCount's middle fallback: no general.parameter_count, but a
// parsed tensor table — the exact element sum wins over the
// one-sig-fig size_label. MoE size labels ("30B-A3B") don't parse, so
// before this fallback those models silently dropped to the gateway's
// 7B default.
func TestMetaParameterCountTensorSumFallback(t *testing.T) {
	m := &Meta{
		KV:          map[string]any{"general.size_label": "30B-A3B"},
		TensorElems: 30_532_122_624,
	}
	require.Equal(t, uint64(30_532_122_624), m.ParameterCount())

	// Explicit parameter_count still wins over the tensor sum (a split
	// model's shard sums to less than the metadata whole).
	m.KV["general.parameter_count"] = uint64(61_000_000_000)
	require.Equal(t, uint64(61_000_000_000), m.ParameterCount())
}

// ActiveParameterCount scales the routed-expert share by
// expert_used_count/expert_count and leaves everything else (dense
// weights, shared experts, router) at full weight. Missing or
// inconsistent inputs fall back to the total — over-predicting is the
// safe direction for cost estimation.
func TestMetaActiveParameterCount(t *testing.T) {
	// Shape mirrors a small MoE: 8_448 dense elems + 262_144 routed
	// expert elems across 128 experts, 8 active per token.
	// active = 8_448 + 262_144 × 8/128 = 24_832.
	moeKV := func() map[string]any {
		return map[string]any{
			"general.architecture":       "qwen3moe",
			"qwen3moe.expert_count":      uint32(128),
			"qwen3moe.expert_used_count": uint32(8),
			"general.parameter_count":    uint64(270_592),
		}
	}
	tests := []struct {
		name string
		kv   map[string]any
		meta Meta
		want uint64
	}{
		{
			name: "MoE scales routed experts by used/count",
			kv:   moeKV(),
			meta: Meta{TensorElems: 270_592, RoutedExpertElems: 262_144},
			want: 24_832,
		},
		{
			name: "dense model (no expert tensors) returns total",
			kv:   map[string]any{"general.architecture": "llama", "general.parameter_count": uint64(1_000)},
			meta: Meta{TensorElems: 1_000},
			want: 1_000,
		},
		{
			name: "expert metadata missing → total",
			kv:   map[string]any{"general.architecture": "qwen3moe", "general.parameter_count": uint64(270_592)},
			meta: Meta{TensorElems: 270_592, RoutedExpertElems: 262_144},
			want: 270_592,
		},
		{
			name: "used >= count (degenerate) → total",
			kv: map[string]any{
				"general.architecture":       "qwen3moe",
				"qwen3moe.expert_count":      uint32(8),
				"qwen3moe.expert_used_count": uint32(8),
				"general.parameter_count":    uint64(270_592),
			},
			meta: Meta{TensorElems: 270_592, RoutedExpertElems: 262_144},
			want: 270_592,
		},
		{
			name: "expert sum >= total (inconsistent size_label) → total",
			kv: map[string]any{
				"general.architecture":       "qwen3moe",
				"qwen3moe.expert_count":      uint32(128),
				"qwen3moe.expert_used_count": uint32(8),
				"general.size_label":         "100K",
			},
			meta: Meta{RoutedExpertElems: 262_144},
			want: 100_000,
		},
		{
			name: "unknown total stays unknown",
			kv:   map[string]any{"general.architecture": "qwen3moe"},
			meta: Meta{RoutedExpertElems: 262_144},
			want: 0,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := tt.meta
			m.KV = tt.kv
			require.Equal(t, tt.want, m.ActiveParameterCount())
		})
	}
}

// ggufBuilder assembles a minimal valid GGUF v3 byte stream: header,
// string/uint32 KV pairs, then a tensor-info table. Only the shapes the
// reader consumes are supported — enough to exercise ReadMeta against
// the real binary layout instead of hand-set struct fields.
type ggufBuilder struct {
	kvKeys  []string
	kvVals  []any // string or uint32
	tensors []ggufTestTensor
}

type ggufTestTensor struct {
	name string
	dims []uint64
}

func (b *ggufBuilder) bytes(t *testing.T) []byte {
	t.Helper()
	var buf bytes.Buffer
	w := func(v any) {
		require.NoError(t, binary.Write(&buf, binary.LittleEndian, v))
	}
	str := func(s string) {
		w(uint64(len(s)))
		buf.WriteString(s)
	}
	buf.WriteString("GGUF")
	w(uint32(3))
	w(uint64(len(b.tensors)))
	w(uint64(len(b.kvKeys)))
	for i, k := range b.kvKeys {
		str(k)
		switch v := b.kvVals[i].(type) {
		case string:
			w(uint32(8)) // typeString
			str(v)
		case uint32:
			w(uint32(4)) // typeUint32
			w(v)
		default:
			t.Fatalf("unsupported kv type %T", v)
		}
	}
	for _, tensor := range b.tensors {
		str(tensor.name)
		w(uint32(len(tensor.dims)))
		for _, d := range tensor.dims {
			w(d)
		}
		w(uint32(0)) // ggml type (F32)
		w(uint64(0)) // data offset
	}
	return buf.Bytes()
}

func writeTempGGUF(t *testing.T, data []byte) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "model.gguf")
	require.NoError(t, os.WriteFile(path, data, 0o600))
	return path
}

// ReadMeta walks the tensor-info table that follows the KV section and
// accumulates total + routed-expert element sums, which feed the MoE
// active-parameter estimate. Shared-expert (_shexp) and router tensors
// must not count as routed experts.
func TestReadMetaTensorSums(t *testing.T) {
	b := &ggufBuilder{
		kvKeys: []string{"general.architecture", "qwen3moe.expert_count", "qwen3moe.expert_used_count"},
		kvVals: []any{"qwen3moe", uint32(128), uint32(8)},
		tensors: []ggufTestTensor{
			{name: "token_embd.weight", dims: []uint64{64, 100}},              // 6_400 dense
			{name: "blk.0.ffn_gate_exps.weight", dims: []uint64{64, 32, 128}}, // 262_144 routed
			{name: "blk.0.ffn_gate_shexp.weight", dims: []uint64{64, 32}},     // 2_048 shared → dense
		},
	}
	path := writeTempGGUF(t, b.bytes(t))

	m, err := ReadMeta(path)
	require.NoError(t, err)
	require.Equal(t, uint64(270_592), m.TensorElems)
	require.Equal(t, uint64(262_144), m.RoutedExpertElems)
	// End-to-end: total from tensor sum, routed share scaled 8/128.
	require.Equal(t, uint64(270_592), m.ParameterCount())
	require.Equal(t, uint64(24_832), m.ActiveParameterCount())
}

// A truncated tensor table must not fail ReadMeta — the KV metadata is
// still good — but the sums must come back zero rather than partial:
// a partial parameter count would silently skew cost prediction.
func TestReadMetaTruncatedTensorTable(t *testing.T) {
	b := &ggufBuilder{
		kvKeys: []string{"general.architecture"},
		kvVals: []any{"llama"},
		tensors: []ggufTestTensor{
			{name: "token_embd.weight", dims: []uint64{64, 100}},
			{name: "blk.0.attn_q.weight", dims: []uint64{64, 64}},
		},
	}
	data := b.bytes(t)
	path := writeTempGGUF(t, data[:len(data)-4]) // cut into the last tensor info

	m, err := ReadMeta(path)
	require.NoError(t, err)
	require.Equal(t, "llama", m.GetString("general.architecture"))
	require.Zero(t, m.TensorElems)
	require.Zero(t, m.RoutedExpertElems)
}

// mmproj (clip-architecture) files surface the vision-encoder shape
// under runtime-agnostic property names; the keys live under
// clip.vision.*, not the <arch>.* pattern the LLM fields use.
func TestSummary_ClipVisionProps(t *testing.T) {
	m := &Meta{KV: map[string]any{
		"general.architecture":           "clip",
		"clip.vision.patch_size":         uint32(16),
		"clip.vision.spatial_merge_size": uint32(2),
	}}
	s := m.Summary()
	require.Equal(t, "16", s["vision_patch_size"])
	require.Equal(t, "2", s["vision_merge_size"])

	// Non-clip architectures don't grow vision keys.
	llm := &Meta{KV: map[string]any{"general.architecture": "qwen35"}}
	_, ok := llm.Summary()["vision_patch_size"]
	require.False(t, ok)
}
