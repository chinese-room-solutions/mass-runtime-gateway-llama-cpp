// pack writes a .mass archive (a Zip) from runtime.yml + the gateway binary.
//
// Usage:
//
//	pack -binary bin/mass-runtime-llama-cpp[.exe] -manifest manifest/runtime.yml -out dist/mass-runtime-llama-cpp.mass
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

	if err := addFile(w, manifestPath, "runtime.yml", 0o644); err != nil {
		return err
	}
	if err := addFile(w, binaryPath, "bin/"+filepath.Base(binaryPath), 0o755); err != nil {
		return err
	}
	return nil
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
