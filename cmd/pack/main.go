// pack writes a .mass archive (a Zip) from runtime.yml + the gateway binary.
//
// Usage:
//
//	pack -binary bin/mass-runtime-gateway-llama-cpp[.exe] -manifest manifest/runtime.yml -out dist/mass-runtime-gateway-llama-cpp.mass
//
// Lives under cmd/ so `go run` from the Makefile picks it up without a
// global PATH dependency on the system zip(1).
package main

import (
	"archive/zip"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"

	"gopkg.in/yaml.v3"
)

func main() {
	binary := flag.String("binary", "", "path to the gateway binary")
	manifest := flag.String("manifest", "", "path to runtime.yml")
	out := flag.String("out", "", "output .mass archive path")
	flag.Parse()
	if *binary == "" || *manifest == "" || *out == "" {
		log.Fatal("usage: pack -binary <path> -manifest <path> -out <path>")
	}

	if err := os.MkdirAll(filepath.Dir(*out), 0o755); err != nil {
		log.Fatalf("creating dist dir: %v", err)
	}
	if err := buildArchive(*out, *binary, *manifest); err != nil {
		log.Fatalf("packing: %v", err)
	}
	fmt.Printf("    Built: %s\n", *out)
}

func buildArchive(outPath, binaryPath, manifestPath string) error {
	if err := os.RemoveAll(outPath); err != nil {
		return fmt.Errorf("removing previous archive: %w", err)
	}
	out, err := os.Create(outPath)
	if err != nil {
		return fmt.Errorf("creating archive file: %w", err)
	}
	defer func() { _ = out.Close() }()

	w := zip.NewWriter(out)
	defer func() { _ = w.Close() }()

	// The archived binary keeps its build-time basename — ".exe" on
	// Windows, bare elsewhere — so the manifest's static `binary:`
	// value can't be trusted: it must point at what we actually pack.
	binaryEntry := "bin/" + filepath.Base(binaryPath)
	manifest, err := os.ReadFile(manifestPath)
	if err != nil {
		return fmt.Errorf("reading manifest %s: %w", manifestPath, err)
	}
	manifest, err = rewriteBinaryField(manifest, binaryEntry)
	if err != nil {
		return fmt.Errorf("rewriting manifest %s: %w", manifestPath, err)
	}
	if err := addBytes(w, manifest, "runtime.yml", 0o644); err != nil {
		return err
	}
	return addFile(w, binaryPath, binaryEntry, 0o755)
}

// rewriteBinaryField returns manifestYAML with its top-level `binary:`
// value set to binaryEntry, adding the field when absent. Works on the
// YAML node tree so untouched fields (comments, literal blocks) keep
// their formatting.
func rewriteBinaryField(manifestYAML []byte, binaryEntry string) ([]byte, error) {
	var doc yaml.Node
	if err := yaml.Unmarshal(manifestYAML, &doc); err != nil {
		return nil, fmt.Errorf("parsing manifest: %w", err)
	}
	if doc.Kind != yaml.DocumentNode || len(doc.Content) == 0 || doc.Content[0].Kind != yaml.MappingNode {
		return nil, fmt.Errorf("manifest is not a YAML mapping")
	}
	m := doc.Content[0]
	found := false
	for i := 0; i+1 < len(m.Content); i += 2 {
		if m.Content[i].Value == "binary" {
			m.Content[i+1].SetString(binaryEntry)
			found = true
			break
		}
	}
	if !found {
		m.Content = append(m.Content,
			&yaml.Node{Kind: yaml.ScalarNode, Value: "binary"},
			&yaml.Node{Kind: yaml.ScalarNode, Value: binaryEntry},
		)
	}
	out, err := yaml.Marshal(&doc)
	if err != nil {
		return nil, fmt.Errorf("re-encoding manifest: %w", err)
	}
	return out, nil
}

func addFile(w *zip.Writer, src, dst string, mode os.FileMode) error {
	f, err := os.Open(src)
	if err != nil {
		return fmt.Errorf("opening %s: %w", src, err)
	}
	defer func() { _ = f.Close() }()
	hdr := &zip.FileHeader{Name: dst, Method: zip.Deflate}
	hdr.SetMode(mode)
	zf, err := w.CreateHeader(hdr)
	if err != nil {
		return fmt.Errorf("creating zip entry %s: %w", dst, err)
	}
	if _, err := io.Copy(zf, f); err != nil {
		return fmt.Errorf("writing %s into archive: %w", dst, err)
	}
	return nil
}

func addBytes(w *zip.Writer, body []byte, dst string, mode os.FileMode) error {
	hdr := &zip.FileHeader{Name: dst, Method: zip.Deflate}
	hdr.SetMode(mode)
	zf, err := w.CreateHeader(hdr)
	if err != nil {
		return fmt.Errorf("creating zip entry %s: %w", dst, err)
	}
	if _, err := zf.Write(body); err != nil {
		return fmt.Errorf("writing %s into archive: %w", dst, err)
	}
	return nil
}
