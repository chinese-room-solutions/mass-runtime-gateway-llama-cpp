.PHONY: build package clean lint test unittest fmt tidy help

BIN_DIR := bin
DIST_DIR := dist
VERSION := $(shell git describe --tags --always --dirty 2>/dev/null || echo dev)

ifeq ($(OS),Windows_NT)
  BINARY_EXT := .exe
else
  BINARY_EXT :=
endif

BINARY := $(BIN_DIR)/mass-runtime-llama-cpp$(BINARY_EXT)
PACKAGE := $(DIST_DIR)/mass-runtime-llama-cpp.mass

build:
	@mkdir -p $(BIN_DIR)
	@echo "==> Building gateway ($(VERSION))..."
	CGO_ENABLED=0 go build -ldflags '-X main.version=$(VERSION)' -o $(BINARY) ./cmd/mass-runtime-llama-cpp
	@echo "    Built: $(BINARY)"

# Pack the binary + manifest into a Zip-format .mass archive that MASS's
# Runtimes tab understands (see internal/runtimes in the mass repo). Uses
# our own Go-based packer so we don't depend on system zip(1).
package: build
	@go run ./cmd/pack -binary $(BINARY) -manifest manifest/runtime.yml -out $(PACKAGE)

clean:
	rm -rf $(BIN_DIR) $(DIST_DIR)

lint:
	go run github.com/golangci/golangci-lint/v2/cmd/golangci-lint@latest run --timeout 10m ./...

unittest:
	go test ./internal/... -short -count=1

test:
	go test ./internal/... -race -count=1 -timeout 10m

fmt:
	gofmt -s -w .

tidy:
	go mod tidy

help:
	@echo ""
	@echo "  mass-runtime-llama-cpp"
	@echo ""
	@echo "  Targets:"
	@echo "    build     Build the gateway binary"
	@echo "    package   Build + pack into a .mass archive (dist/)"
	@echo "    test      Run tests with -race"
	@echo "    lint      Run golangci-lint"
	@echo "    fmt       Format Go code"
	@echo "    tidy      go mod tidy"
	@echo "    clean     Remove build outputs"
	@echo ""
