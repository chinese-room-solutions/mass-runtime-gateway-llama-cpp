.PHONY: build package clean lint test unittest fmt tidy gen help

BIN_DIR := bin
DIST_DIR := dist

ifeq ($(OS),Windows_NT)
  BINARY_EXT := .exe
else
  BINARY_EXT :=
endif

BINARY := $(BIN_DIR)/mass-runtime-gateway-llama-cpp$(BINARY_EXT)
PACKAGE := $(DIST_DIR)/mass-runtime-gateway-llama-cpp.mass

# The gateway reads its version from runtime.yml at runtime — see
# cmd/mass-runtime-gateway-llama-cpp/main.go and mass-sdk/manifest. We therefore
# don't bake a -X main.version=... build flag.
build:
	@mkdir -p $(BIN_DIR)
	@echo "==> Building gateway..."
	CGO_ENABLED=0 go build -o $(BINARY) ./cmd/mass-runtime-gateway-llama-cpp
	@echo "    Built: $(BINARY)"

# Pack the binary + manifest into a Zip-format .mass archive that MASS's
# Runtimes tab understands (see internal/runtimes in the mass repo). Uses
# our own Go-based packer so we don't depend on system zip(1).
package: build
	@go run ./cmd/pack -binary $(BINARY) -manifest manifest/runtime.yml -out $(PACKAGE)

clean:
	rm -rf $(BIN_DIR) $(DIST_DIR)

lint:
	@# golangci-lint must be built with a toolchain >= the repo's go directive or it refuses to load.
	GOTOOLCHAIN=go$$(go list -m -f '{{.GoVersion}}') go run github.com/golangci/golangci-lint/v2/cmd/golangci-lint@v2.5.0 run --timeout 10m ./...

unittest:
	go test ./... -short -count=1

test:
	go test ./... -race -count=1 -timeout 10m

fmt:
	gofmt -s -w .

tidy:
	go mod tidy

# Regenerate Go bindings for proto/llama_cpp/v1/service.proto into
# gen/go/llama_cpp/v1/. The typed gRPC API is consumed by:
#   - this gateway (server)
#   - llama-cpp-rpc-client-go (client)
#
# Requires protoc + protoc-gen-go + protoc-gen-go-grpc on PATH.
gen:
	protoc -I=proto --go_out=gen/go --go_opt=paths=source_relative proto/llama_cpp/v1/service.proto
	protoc -I=proto --go-grpc_out=gen/go --go-grpc_opt=paths=source_relative proto/llama_cpp/v1/service.proto

help:
	@echo ""
	@echo "  mass-runtime-gateway-llama-cpp"
	@echo ""
	@echo "  Targets:"
	@echo "    build     Build the gateway binary"
	@echo "    package   Build + pack into a .mass archive (dist/)"
	@echo "    test      Run tests with -race"
	@echo "    lint      Run golangci-lint"
	@echo "    fmt       Format Go code"
	@echo "    tidy      go mod tidy"
	@echo "    gen       Regenerate proto Go bindings (gen/go/llama_cpp/v1/)"
	@echo "    clean     Remove build outputs"
	@echo ""
