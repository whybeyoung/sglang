package core

import (
	"context"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/sglang/sglang-router-go/pkg/proto"
	"go.uber.org/zap"
	"google.golang.org/grpc"
	"google.golang.org/grpc/connectivity"
	"google.golang.org/grpc/credentials/insecure"
)

// HealthChecker performs health checks on workers
// Similar to Rust's health check implementation
type HealthChecker struct {
	logger       *zap.Logger
	httpClient   *http.Client
	timeout      time.Duration
	failureCount map[string]int
	mu           sync.RWMutex
}

// NewHealthChecker creates a new health checker
func NewHealthChecker(logger *zap.Logger, timeout time.Duration) *HealthChecker {
	return &HealthChecker{
		logger:       logger,
		httpClient:   &http.Client{Timeout: timeout},
		timeout:      timeout,
		failureCount: make(map[string]int),
	}
}

// CheckHealth performs a health check on a worker
func (hc *HealthChecker) CheckHealth(ctx context.Context, worker Worker) error {
	switch worker.ConnectionMode() {
	case ConnectionModeHTTP:
		return hc.checkHTTPHealth(ctx, worker)
	case ConnectionModeGRPC:
		return hc.checkGRPCHealth(ctx, worker)
	default:
		return fmt.Errorf("unknown connection mode: %v", worker.ConnectionMode())
	}
}

// checkHTTPHealth performs HTTP health check
// Similar to Rust http_health_check
func (hc *HealthChecker) checkHTTPHealth(ctx context.Context, worker Worker) error {
	url := worker.URL()

	// Remove protocol prefix if present
	cleanURL := strings.TrimPrefix(url, "http://")
	cleanURL = strings.TrimPrefix(cleanURL, "https://")

	// Build health check URL
	healthURL := fmt.Sprintf("http://%s/health", cleanURL)

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, healthURL, nil)
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	// Add API key if available
	if apiKey := worker.APIKey(); apiKey != nil && *apiKey != "" {
		req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", *apiKey))
	}

	resp, err := hc.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("health check request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusOK {
		return nil
	}

	return fmt.Errorf("health check returned status %d", resp.StatusCode)
}

// checkGRPCHealth performs gRPC health check using HealthCheck RPC
// Similar to Rust grpc_health_check
func (hc *HealthChecker) checkGRPCHealth(ctx context.Context, worker Worker) error {
	url := worker.URL()

	// Extract gRPC address
	grpcAddr, err := extractGRPCAddress(url)
	if err != nil {
		return fmt.Errorf("invalid gRPC URL: %w", err)
	}

	// Create gRPC connection with timeout
	dialCtx, cancel := context.WithTimeout(ctx, hc.timeout)
	defer cancel()

	conn, err := grpc.DialContext(dialCtx, grpcAddr,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithBlock(),
	)
	if err != nil {
		return fmt.Errorf("failed to connect: %w", err)
	}
	defer conn.Close()

	// Check connection state first (quick check)
	state := conn.GetState()
	if state != connectivity.Ready && state != connectivity.Idle {
		return fmt.Errorf("connection not ready: %v", state)
	}

	// Call HealthCheck RPC using proto client
	client := proto.NewSglangSchedulerClient(conn)

	healthCtx, cancel := context.WithTimeout(ctx, hc.timeout)
	defer cancel()

	resp, err := client.HealthCheck(healthCtx, &proto.HealthCheckRequest{})
	if err != nil {
		return fmt.Errorf("health check RPC failed: %w", err)
	}

	if resp != nil && resp.Healthy {
		hc.logger.Debug("gRPC health check passed",
			zap.String("url", url),
			zap.String("state", state.String()),
		)
		return nil
	}

	return fmt.Errorf("health check returned unhealthy")
}

// extractGRPCAddress extracts gRPC address from URL
func extractGRPCAddress(url string) (string, error) {
	if strings.HasPrefix(url, "grpc://") {
		return url[7:], nil
	}
	if len(url) > 0 && !strings.Contains(url, "://") {
		return url, nil
	}
	return "", fmt.Errorf("invalid gRPC URL format: %s", url)
}
