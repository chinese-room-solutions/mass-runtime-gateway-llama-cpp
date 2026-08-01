package gateway

import (
	"testing"

	hf "github.com/chinese-room-solutions/mass-sdk/huggingface"
	"github.com/stretchr/testify/require"
)

// Repos often ship both mmproj-f16 and mmproj-f32; the API's listing
// order isn't a contract, so the companion pick must not depend on it:
// f16 wins whenever present, otherwise the first candidate is used.
func TestPickMmprojCompanion(t *testing.T) {
	tests := []struct {
		name     string
		files    []hf.GGUFFile
		primary  string
		wantName string
		wantSize int64
	}{
		{
			name: "f16 preferred over f32 listed first",
			files: []hf.GGUFFile{
				{Filename: "model-Q4_K_M.gguf", SizeBytes: 100},
				{Filename: "mmproj-model-f32.gguf", SizeBytes: 32},
				{Filename: "mmproj-model-f16.gguf", SizeBytes: 16},
			},
			primary:  "model-Q4_K_M.gguf",
			wantName: "mmproj-model-f16.gguf",
			wantSize: 16,
		},
		{
			name: "f16 preferred when listed first too",
			files: []hf.GGUFFile{
				{Filename: "mmproj-model-f16.gguf", SizeBytes: 16},
				{Filename: "mmproj-model-f32.gguf", SizeBytes: 32},
				{Filename: "model-Q4_K_M.gguf", SizeBytes: 100},
			},
			primary:  "model-Q4_K_M.gguf",
			wantName: "mmproj-model-f16.gguf",
			wantSize: 16,
		},
		{
			name: "f16 match is case-insensitive",
			files: []hf.GGUFFile{
				{Filename: "mmproj-model-f32.gguf", SizeBytes: 32},
				{Filename: "mmproj-model-F16.gguf", SizeBytes: 16},
			},
			primary:  "model-Q4_K_M.gguf",
			wantName: "mmproj-model-F16.gguf",
			wantSize: 16,
		},
		{
			name: "only f32 falls back to first candidate",
			files: []hf.GGUFFile{
				{Filename: "model-Q4_K_M.gguf", SizeBytes: 100},
				{Filename: "mmproj-model-f32.gguf", SizeBytes: 32},
			},
			primary:  "model-Q4_K_M.gguf",
			wantName: "mmproj-model-f32.gguf",
			wantSize: 32,
		},
		{
			name: "no f16 picks first of several candidates",
			files: []hf.GGUFFile{
				{Filename: "mmproj-model-f32.gguf", SizeBytes: 32},
				{Filename: "mmproj-model-bf32.gguf", SizeBytes: 33},
			},
			primary:  "model-Q4_K_M.gguf",
			wantName: "mmproj-model-f32.gguf",
			wantSize: 32,
		},
		{
			name: "no mmproj returns empty",
			files: []hf.GGUFFile{
				{Filename: "model-Q4_K_M.gguf", SizeBytes: 100},
				{Filename: "model-Q8_0.gguf", SizeBytes: 200},
			},
			primary:  "model-Q4_K_M.gguf",
			wantName: "",
			wantSize: -1,
		},
		{
			name: "primary itself is never picked as companion",
			files: []hf.GGUFFile{
				{Filename: "mmproj-model-f16.gguf", SizeBytes: 16},
			},
			primary:  "mmproj-model-f16.gguf",
			wantName: "",
			wantSize: -1,
		},
		{
			name:     "empty listing returns empty",
			files:    nil,
			primary:  "model.gguf",
			wantName: "",
			wantSize: -1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			name, size := pickMmprojCompanion(tt.files, tt.primary)
			require.Equal(t, tt.wantName, name)
			require.Equal(t, tt.wantSize, size)
		})
	}
}

// Filenames may live in repo subfolders; the "/" separators must
// survive as path separators while each segment's special characters
// are still escaped.
func TestHFFileURL(t *testing.T) {
	tests := []struct {
		name     string
		repoID   string
		filename string
		want     string
	}{
		{
			name:     "plain filename",
			repoID:   "org/model-GGUF",
			filename: "model-Q4_K_M.gguf",
			want:     "https://huggingface.co/org/model-GGUF/resolve/main/model-Q4_K_M.gguf",
		},
		{
			name:     "subfolder keeps slash as separator",
			repoID:   "org/model-GGUF",
			filename: "Q4_K_M/model-Q4_K_M-00001-of-00002.gguf",
			want:     "https://huggingface.co/org/model-GGUF/resolve/main/Q4_K_M/model-Q4_K_M-00001-of-00002.gguf",
		},
		{
			name:     "nested subfolders",
			repoID:   "org/model-GGUF",
			filename: "a/b/model.gguf",
			want:     "https://huggingface.co/org/model-GGUF/resolve/main/a/b/model.gguf",
		},
		{
			name:     "special characters escaped per segment",
			repoID:   "org/model-GGUF",
			filename: "sub dir/model 100%.gguf",
			want:     "https://huggingface.co/org/model-GGUF/resolve/main/sub%20dir/model%20100%25.gguf",
		},
		{
			name:     "plus and hash escaped",
			repoID:   "org/model-GGUF",
			filename: "q/model#v2+fp16.gguf",
			want:     "https://huggingface.co/org/model-GGUF/resolve/main/q/model%23v2+fp16.gguf",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			require.Equal(t, tt.want, hfFileURL(tt.repoID, tt.filename))
		})
	}
}
