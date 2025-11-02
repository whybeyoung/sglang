package grpc

import (
	"context"

	"github.com/sglang/sglang-router-go/internal/core"
	"github.com/sglang/sglang-router-go/pkg/proto"
	"go.uber.org/zap"
	"google.golang.org/grpc"
)

// SglangSchedulerClientWrapper wraps the proto client for easier use
type SglangSchedulerClientWrapper struct {
	client proto.SglangSchedulerClient
	conn   *grpc.ClientConn
	logger *zap.Logger
}

// NewSglangSchedulerClientWrapper creates a new client wrapper
func NewSglangSchedulerClientWrapper(conn *grpc.ClientConn, logger *zap.Logger) *SglangSchedulerClientWrapper {
	return &SglangSchedulerClientWrapper{
		client: proto.NewSglangSchedulerClient(conn),
		conn:   conn,
		logger: logger,
	}
}

// Generate starts a Generate RPC and returns a stream
func (c *SglangSchedulerClientWrapper) Generate(ctx context.Context, req *proto.GenerateRequest) (proto.SglangScheduler_GenerateClient, error) {
	return c.client.Generate(ctx, req)
}

// HealthCheck performs a health check
func (c *SglangSchedulerClientWrapper) HealthCheck(ctx context.Context) error {
	_, err := c.client.HealthCheck(ctx, &proto.HealthCheckRequest{})
	return err
}

// GetModelInfo gets model information including tokenizer path
func (c *SglangSchedulerClientWrapper) GetModelInfo(ctx context.Context) (*proto.GetModelInfoResponse, error) {
	return c.client.GetModelInfo(ctx, &proto.GetModelInfoRequest{})
}

// GetTokenizerInfo gets tokenizer files and configuration content directly
func (c *SglangSchedulerClientWrapper) GetTokenizerInfo(ctx context.Context, requestedFiles []string) (*proto.GetTokenizerInfoResponse, error) {
	req := &proto.GetTokenizerInfoRequest{
		RequestedFiles: requestedFiles,
	}
	return c.client.GetTokenizerInfo(ctx, req)
}

// Close closes the connection (not needed, connection managed by pool)
func (c *SglangSchedulerClientWrapper) Close() error {
	// Connection is managed by ClientPool, don't close here
	return nil
}

// GetClientWrapper gets or creates a proto client wrapper for a worker
func (p *ClientPool) GetClientWrapper(ctx context.Context, worker core.Worker) (*SglangSchedulerClientWrapper, error) {
	conn, err := p.GetClient(ctx, worker)
	if err != nil {
		return nil, err
	}
	return NewSglangSchedulerClientWrapper(conn, p.logger), nil
}
