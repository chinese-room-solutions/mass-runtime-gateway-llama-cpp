// mass-runtime-llama-cpp is a hashicorp/go-plugin subprocess MASS launches
// to terminate inference traffic for the llama-cpp runtime kind.
//
// Lifecycle:
//  1. MASS execs this binary, completing the go-plugin handshake.
//  2. MASS dispenses the "runtime_gateway" plugin and calls Init.
//  3. We register our gRPC server with the gateway service.
//
// Inference HTTP requests land here via MASS's /mass.llama-cpp.* HTTP proxy
// → RuntimeGateway.HandleRequest streaming gRPC. We dispatch based on the
// HTTP path (typed API, OpenAI-compat) and forward jobs to the worker via
// the MASS scheduler callback.
package main

import (
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/KernelPryanic/golog"
	"github.com/chinese-room-solutions/mass-runtime-llama-cpp/internal/gateway"
	"github.com/hashicorp/go-plugin"
	"github.com/rs/zerolog"
)

// version is set at build time via -ldflags "-X main.version=...".
var version = "dev"

// runtimeName is the stable identifier for this gateway. Must match
// runtime.yml and the URL prefix MASS uses (`/mass.llama-cpp.*`).
const runtimeName = "llama-cpp"

// displayName is used by MASS's Runtimes tab when InitResponse.display_name
// is empty.
const displayName = "llama.cpp"

func main() {
	showVersion := flag.Bool("version", false, "Print version and exit")
	flag.Parse()
	if *showVersion {
		fmt.Println("mass-runtime-llama-cpp", version)
		return
	}

	// All log output goes to stderr. hashicorp/go-plugin captures stderr
	// and routes it back to MASS's logger via the hclog adapter MASS
	// installed when launching us.
	logger := golog.New(false, io.MultiWriter(os.Stderr))
	zerolog.SetGlobalLevel(zerolog.DebugLevel) // overridden by Init.LogLevel

	plugin.Serve(&plugin.ServeConfig{
		HandshakeConfig: gateway.Handshake,
		Plugins: map[string]plugin.Plugin{
			gateway.PluginName: gateway.NewPlugin(gateway.PluginParams{
				RuntimeName: runtimeName,
				Version:     version,
				DisplayName: displayName,
				Logger:      logger,
			}),
		},
		GRPCServer: plugin.DefaultGRPCServer,
		Logger:     gateway.HCLogAdapter(logger),
	})
}
