package main

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/sglang/sglang-router-go/internal/config"
	"github.com/sglang/sglang-router-go/internal/core"
	"github.com/sglang/sglang-router-go/internal/router"
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
	// TODO: Parse worker URLs and register them
	// For now, this is a placeholder
	_ = workerRegistry

	// Create gRPC router
	grpcRouter, err := router.NewGrpcRouter(
		workerRegistry,
		nil, // TODO: Create policy registry
		nil, // TODO: Create tokenizer
		nil, // TODO: Create tool parser factory
		nil, // TODO: Create reasoning parser factory
		logger,
	)
	if err != nil {
		logger.Fatal("Failed to create gRPC router", zap.Error(err))
	}

	_ = grpcRouter

	// Start health checker
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	workerRegistry.StartHealthChecker(ctx, 60*time.Second)
	defer workerRegistry.StopHealthChecker()

	// TODO: Start HTTP server for API endpoints
	// TODO: Start gRPC server if needed
	// TODO: Start Prometheus metrics server

	logger.Info("SGLang Router started",
		zap.String("host", cfg.Host),
		zap.Uint16("port", cfg.Port),
	)

	// Wait for interrupt signal
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)

	<-sigChan
	logger.Info("Shutting down...")

	// Cleanup
	cancel()
	workerRegistry.StopHealthChecker()

	logger.Info("Shutdown complete")
}
