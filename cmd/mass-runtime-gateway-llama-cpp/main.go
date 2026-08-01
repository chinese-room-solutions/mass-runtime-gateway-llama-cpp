// mass-runtime-gateway-llama-cpp is a hashicorp/go-plugin subprocess MASS launches
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
	"github.com/chinese-room-solutions/mass-runtime-gateway-llama-cpp/internal/gateway"
	sdkmanifest "github.com/chinese-room-solutions/mass-sdk/manifest"
	"github.com/hashicorp/go-plugin"
	"github.com/rs/zerolog"
)

func main() {
	showVersion := flag.Bool("version", false, "Print version and exit")
	flag.Parse()

	// runtime.yml is the single source of truth for runtime_name, version,
	// display name, and description. MASS extracts the .mass package to
	// <runtimesDir>/<runtimeName>/, with runtime.yml at the root and our
	// binary in bin/. The SDK helper finds runtime.yml relative to our own
	// executable so we don't need MASS to pass it via env or flags.
	mf, err := sdkmanifest.LoadAdjacentToBinary()
	if err != nil {
		fmt.Fprintln(os.Stderr, "mass-runtime-gateway-llama-cpp: failed to load runtime.yml:", err)
		os.Exit(1)
	}

	if *showVersion {
		fmt.Println("mass-runtime-gateway-llama-cpp", mf.Version)
		return
	}

	// All log output goes to stderr. hashicorp/go-plugin captures stderr
	// and routes it back to MASS's logger via the hclog adapter MASS
	// installed when launching us.
	logger := golog.New(false, io.MultiWriter(os.Stderr))
	zerolog.SetGlobalLevel(zerolog.InfoLevel) // overridden by Init.LogLevel

	plugin.Serve(&plugin.ServeConfig{
		HandshakeConfig: gateway.Handshake,
		Plugins: map[string]plugin.Plugin{
			gateway.PluginName: gateway.NewPlugin(gateway.PluginParams{
				RuntimeName: mf.RuntimeName,
				Version:     mf.Version,
				DisplayName: mf.DisplayName,
				Description: mf.Description,
				Logger:      logger,
			}),
		},
		GRPCServer: plugin.DefaultGRPCServer,
		Logger:     gateway.HCLogAdapter(logger),
	})
}
