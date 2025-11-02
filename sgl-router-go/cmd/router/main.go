package main

import (
	"context"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/sglang/sglang-router-go/internal/config"
	"github.com/sglang/sglang-router-go/internal/core"
	"github.com/sglang/sglang-router-go/internal/policy"
	"github.com/sglang/sglang-router-go/internal/router"
	"github.com/sglang/sglang-router-go/internal/server"
	"go.uber.org/zap"
)

func main() {
	// Initialize logger
	logger, err := zap.NewProduction()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to initialize logger: %v\n", err)
		os.Exit(1)
	}
	defer logger.Sync()

	// Load configuration
	cfg, err := config.LoadFromFlags()
	if err != nil {
		logger.Fatal("Failed to load configuration", zap.Error(err))
	}

	// Create worker registry
	workerRegistry := core.NewWorkerRegistry(logger)

	// Register initial workers from configuration
	for _, workerURL := range cfg.WorkerURLs {
		parsedURL, err := url.Parse(workerURL)
		if err != nil {
			logger.Warn("Invalid worker URL, skipping",
				zap.String("url", workerURL),
				zap.Error(err),
			)
			continue
		}

		// Determine connection mode
		var connMode core.ConnectionMode
		if strings.HasPrefix(workerURL, "grpc://") {
			connMode = core.ConnectionModeGRPC
		} else {
			connMode = core.ConnectionModeHTTP
		}

		// Extract model ID from URL or use default
		modelID := parsedURL.Query().Get("model")
		if modelID == "" {
			modelID = "default"
		}

		// Create worker metadata
		metadata := &core.WorkerMetadata{
			URL:            workerURL,
			ModelID:        modelID,
			WorkerType:     core.WorkerTypeRegular,
			ConnectionMode: connMode,
			Priority:       50,
			Cost:           1.0,
			Labels:         make(map[string]string),
		}

		// Create and register worker
		worker := core.NewBasicWorker(metadata)
		workerRegistry.Register(worker)

		logger.Info("Registered worker",
			zap.String("url", workerURL),
			zap.String("model_id", modelID),
			zap.String("connection_mode", string(connMode)),
		)
	}

	// Create policy registry
	var defaultPolicy policy.Policy
	switch cfg.Policy {
	case "random":
		defaultPolicy = policy.NewRandomPolicy()
	case "round_robin":
		defaultPolicy = policy.NewRoundRobinPolicy()
	default:
		// Default to round_robin if unknown policy
		logger.Warn("Unknown policy, using round_robin", zap.String("policy", cfg.Policy))
		defaultPolicy = policy.NewRoundRobinPolicy()
	}
	policyRegistry := policy.NewPolicyRegistry(defaultPolicy)

	// Create gRPC router
	grpcRouter, err := router.NewGrpcRouter(
		workerRegistry,
		policyRegistry,
		nil, // TODO: Create tokenizer when tokenizer library is integrated
		nil, // TODO: Create tool parser factory
		nil, // TODO: Create reasoning parser factory
		logger,
	)
	if err != nil {
		logger.Fatal("Failed to create gRPC router", zap.Error(err))
	}

	// Start health checker
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	workerRegistry.StartHealthChecker(ctx, 60*time.Second)
	defer workerRegistry.StopHealthChecker()

	// Start HTTP server
	httpServer := server.NewHTTPServer(
		grpcRouter,
		cfg.Host,
		cfg.Port,
		logger,
	)

	// Start HTTP server in a goroutine
	serverErrChan := make(chan error, 1)
	go func() {
		if err := httpServer.Start(); err != nil && err != http.ErrServerClosed {
			serverErrChan <- err
		}
	}()

	// TODO: Start Prometheus metrics server if enabled

	logger.Info("SGLang Router started",
		zap.String("host", cfg.Host),
		zap.Uint16("port", cfg.Port),
	)

	// Wait for interrupt signal
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)

	select {
	case sig := <-sigChan:
		logger.Info("Received shutdown signal", zap.String("signal", sig.String()))
	case err := <-serverErrChan:
		logger.Fatal("HTTP server error", zap.Error(err))
	}

	logger.Info("Shutting down...")

	// Graceful shutdown
	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer shutdownCancel()

	// Shutdown HTTP server
	if err := httpServer.Shutdown(shutdownCtx); err != nil {
		logger.Warn("Error shutting down HTTP server", zap.Error(err))
	}

	// Stop health checker
	cancel()
	workerRegistry.StopHealthChecker()

	logger.Info("Shutdown complete")
}
