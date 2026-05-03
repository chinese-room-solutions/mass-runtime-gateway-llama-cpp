// Package gateway implements the runtime gateway side of the MASS
// gateway-plugin contract for llama.cpp.
package gateway

import (
	"context"
	"errors"
	"fmt"

	gatewaypb "github.com/chinese-room-solutions/mass-proto/gen/go/gateway"
	"github.com/hashicorp/go-plugin"
	"github.com/rs/zerolog"
	"google.golang.org/grpc"
)

// Handshake mirrors mass/internal/runtimes.Handshake — both sides MUST agree
// on cookie + value or hashicorp/go-plugin refuses the connection.
var Handshake = plugin.HandshakeConfig{
	ProtocolVersion:  1,
	MagicCookieKey:   "MASS_RUNTIME_PLUGIN",
	MagicCookieValue: "1f2c0c1e-mass-runtime-gateway",
}

// PluginName is the dispense key MASS uses. Must match the host-side
// constant in mass/internal/runtimes.PluginName.
const PluginName = "runtime_gateway"

// PluginParams configures a Plugin instance.
type PluginParams struct {
	RuntimeKind string
	Version     string
	DisplayName string
	Logger      zerolog.Logger
}

// Plugin is the go-plugin descriptor for the gateway. It owns one in-process
// [Gateway] (constructed lazily during the first GRPCServer call so we have
// access to the broker — needed to dial back to MASS).
type Plugin struct {
	plugin.Plugin
	params PluginParams
}

// NewPlugin builds the gateway plugin descriptor.
func NewPlugin(params PluginParams) *Plugin {
	return &Plugin{params: params}
}

// GRPCServer is invoked by go-plugin on the worker side (i.e. inside this
// subprocess). It registers our [Gateway] as the RuntimeGateway service and
// stashes the broker so we can dial MASS's MassScheduler when Init arrives.
func (p *Plugin) GRPCServer(broker *plugin.GRPCBroker, srv *grpc.Server) error {
	if broker == nil {
		return errors.New("gateway: GRPCBroker missing")
	}
	gw := newGateway(p.params, broker)
	gatewaypb.RegisterRuntimeGatewayServer(srv, gw)
	return nil
}

// GRPCClient is unused on the gateway (worker) side — MASS never serves
// the gateway service back to itself.
func (p *Plugin) GRPCClient(_ context.Context, _ *plugin.GRPCBroker, _ *grpc.ClientConn) (any, error) {
	return nil, fmt.Errorf("gateway: GRPCClient not used in plugin process")
}
