package gateway

import (
	"context"
	"testing"

	gatewaypb "github.com/chinese-room-solutions/mass-proto/gen/go/gateway"
	"github.com/rs/zerolog"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// Every model RPC must refuse cleanly (FailedPrecondition) when called
// before Init has populated modelsDir/cache — instead of dereferencing
// a nil cache or reading half-initialised state.
func TestModelRPCs_RefuseBeforeInit(t *testing.T) {
	g := &Gateway{logger: zerolog.Nop()}
	ctx := context.Background()

	tests := []struct {
		name string
		call func() error
	}{
		{
			name: "ListGroups",
			call: func() error {
				_, err := g.ListGroups(ctx, &gatewaypb.GatewayListGroupsRequest{})
				return err
			},
		},
		{
			name: "PlanLocalImport",
			call: func() error {
				_, err := g.PlanLocalImport(ctx, &gatewaypb.PlanLocalImportRequest{SrcPath: "/tmp/a.gguf", GroupName: "G"})
				return err
			},
		},
		{
			name: "PlanRemoteImport",
			call: func() error {
				_, err := g.PlanRemoteImport(ctx, &gatewaypb.PlanRemoteImportRequest{RepoId: "org/repo", Filename: "a.gguf", GroupName: "G"})
				return err
			},
		},
		{
			name: "PlanDelete",
			call: func() error {
				_, err := g.PlanDelete(ctx, &gatewaypb.PlanDeleteRequest{Id: "some-group"})
				return err
			},
		},
		{
			name: "RenameGroup",
			call: func() error {
				_, err := g.RenameGroup(ctx, &gatewaypb.RenameGroupRequest{Id: "some-group", NewName: "New"})
				return err
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.call()
			require.Error(t, err)
			require.Equal(t, codes.FailedPrecondition, status.Code(err))
		})
	}
}
