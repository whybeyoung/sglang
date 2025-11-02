package grpc

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/sglang/sglang-router-go/internal/core"
	"go.uber.org/zap"
	"google.golang.org/grpc"
	"google.golang.org/grpc/connectivity"
	"google.golang.org/grpc/credentials/insecure"
)

// ClientPool manages gRPC client connections
// Similar to Rust's gRPC client management
type ClientPool struct {
	mu       sync.RWMutex
	clients  map[string]*grpc.ClientConn
	logger   *zap.Logger
	dialOpts []grpc.DialOption
}

// NewClientPool creates a new gRPC client pool
func NewClientPool(logger *zap.Logger) *ClientPool {
	// Set maximum message size to 100MB to handle large tokenizer files
	// Default is 4MB which is too small for tokenizer.json files
	maxMsgSize := 100 * 1024 * 1024 // 100MB

	return &ClientPool{
		clients: make(map[string]*grpc.ClientConn),
		logger:  logger,
		dialOpts: []grpc.DialOption{
			grpc.WithTransportCredentials(insecure.NewCredentials()),
			grpc.WithBlock(), // Wait for connection
			grpc.WithDefaultCallOptions(
				grpc.MaxCallRecvMsgSize(maxMsgSize),
				grpc.MaxCallSendMsgSize(maxMsgSize),
			),
		},
		// Note: grpc.WithBlock() will block until connection is established or context times out
	}
}

// GetClient gets or creates a gRPC client connection for a worker
func (p *ClientPool) GetClient(ctx context.Context, worker core.Worker) (*grpc.ClientConn, error) {
	url := worker.URL()

	p.mu.RLock()
	conn, exists := p.clients[url]
	p.mu.RUnlock()

	if exists {
		// Check if connection is still valid
		state := conn.GetState()
		if state == connectivity.Ready || state == connectivity.Idle {
			return conn, nil
		}
		// Connection is not ready, remove it and create a new one
		p.mu.Lock()
		delete(p.clients, url)
		p.mu.Unlock()
		conn.Close()
	}

	// Extract gRPC address from URL
	// Format: grpc://host:port
	grpcAddr, err := extractGRPCAddress(url)
	if err != nil {
		return nil, fmt.Errorf("invalid gRPC URL: %w", err)
	}

	// Create new connection
	p.logger.Info("Creating new gRPC connection",
		zap.String("url", url),
		zap.String("extracted_address", grpcAddr),
	)

	// Create context with timeout for dialing
	// Default dial timeout is 15 seconds
	dialTimeout := 15 * time.Second

	// Check if parent context has a deadline
	deadline, hasDeadline := ctx.Deadline()
	if hasDeadline && ctx != context.Background() && ctx != context.TODO() {
		// Parent context has a deadline, use the smaller of:
		// - remaining time until parent deadline
		// - default dial timeout
		remaining := time.Until(deadline)
		if remaining < dialTimeout {
			dialTimeout = remaining
			if dialTimeout <= 0 {
				return nil, fmt.Errorf("parent context already expired")
			}
		}
	}

	// Create dial context with timeout
	dialCtx, cancel := context.WithTimeout(ctx, dialTimeout)
	defer cancel()

	p.logger.Info("Dialing gRPC server",
		zap.String("address", grpcAddr),
		zap.Duration("timeout", dialTimeout),
		zap.String("original_url", url),
	)

	newConn, err := grpc.DialContext(dialCtx, grpcAddr, p.dialOpts...)
	if err != nil {
		p.logger.Error("Failed to dial gRPC server",
			zap.String("address", grpcAddr),
			zap.Duration("timeout", dialTimeout),
			zap.Error(err),
		)
		return nil, fmt.Errorf("failed to dial gRPC (address: %s, timeout: %v): %w", grpcAddr, dialTimeout, err)
	}

	p.logger.Info("Successfully connected to gRPC server",
		zap.String("address", grpcAddr),
		zap.String("state", newConn.GetState().String()),
	)

	// Store connection
	p.mu.Lock()
	p.clients[url] = newConn
	p.mu.Unlock()

	return newConn, nil
}

// Close closes all connections in the pool
func (p *ClientPool) Close() error {
	p.mu.Lock()
	defer p.mu.Unlock()

	var errs []error
	for url, conn := range p.clients {
		if err := conn.Close(); err != nil {
			errs = append(errs, fmt.Errorf("failed to close connection to %s: %w", url, err))
		}
	}
	p.clients = make(map[string]*grpc.ClientConn)

	if len(errs) > 0 {
		return fmt.Errorf("errors closing connections: %v", errs)
	}
	return nil
}

// RemoveClient removes a client connection from the pool
func (p *ClientPool) RemoveClient(url string) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if conn, exists := p.clients[url]; exists {
		conn.Close()
		delete(p.clients, url)
	}
}

// extractGRPCAddress extracts the gRPC address from a URL
// Converts "grpc://host:port" to "host:port"
func extractGRPCAddress(url string) (string, error) {
	// Remove "grpc://" prefix if present
	if len(url) > 7 && url[:7] == "grpc://" {
		return url[7:], nil
	}
	// If no prefix, assume it's already the address
	if len(url) > 0 {
		return url, nil
	}
	return "", fmt.Errorf("empty URL")
}
