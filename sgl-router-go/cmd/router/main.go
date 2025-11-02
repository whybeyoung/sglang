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
	"github.com/sglang/sglang-router-go/internal/grpc"
	"github.com/sglang/sglang-router-go/internal/policy"
	"github.com/sglang/sglang-router-go/internal/router"
	"github.com/sglang/sglang-router-go/internal/server"
	"github.com/sglang/sglang-router-go/internal/tokenizer"
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
		} else if strings.HasPrefix(workerURL, "http://") || strings.HasPrefix(workerURL, "https://") {
			connMode = core.ConnectionModeHTTP
		} else {
			// Default to HTTP if no protocol prefix (assume http://)
			connMode = core.ConnectionModeHTTP
			if !strings.HasPrefix(workerURL, "http://") && !strings.HasPrefix(workerURL, "https://") {
				workerURL = "http://" + workerURL
			}
		}

		// Extract model ID from URL or use default
		modelID := parsedURL.Query().Get("model")
		if modelID == "" {
			modelID = "default"
		}

		// Normalize worker URL for HTTP mode
		if connMode == core.ConnectionModeHTTP {
			// Ensure http:// prefix
			if !strings.HasPrefix(workerURL, "http://") && !strings.HasPrefix(workerURL, "https://") {
				workerURL = "http://" + workerURL
			}
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
		worker := core.NewBasicWorkerWithLogger(metadata, logger)
		workerRegistry.Register(worker)

		// Perform initial health check synchronously for HTTP workers
		if connMode == core.ConnectionModeHTTP {
			healthCtx, healthCancel := context.WithTimeout(context.Background(), 5*time.Second)
			err := worker.CheckHealth(healthCtx)
			healthCancel()
			if err != nil {
				logger.Warn("Initial health check failed, worker will be checked again",
					zap.String("url", workerURL),
					zap.Error(err),
				)
			} else {
				logger.Info("Initial health check passed",
					zap.String("url", workerURL),
				)
			}
		}

		logger.Info("Registered worker",
			zap.String("url", workerURL),
			zap.String("model_id", modelID),
			zap.String("connection_mode", string(connMode)),
			zap.Bool("healthy", worker.IsHealthy()),
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

	// Create tokenizer if tokenizer path or model path is configured
	// Similar to Rust: tokenizer_path.or_else(|| model_path.clone())
	var tok tokenizer.Tokenizer
	tokenizerPath := ""
	var chatTemplatePath *string

	if cfg.TokenizerPath != nil && *cfg.TokenizerPath != "" {
		tokenizerPath = *cfg.TokenizerPath
	} else if cfg.ModelPath != nil && *cfg.ModelPath != "" {
		tokenizerPath = *cfg.ModelPath
	} else if cfg.GRPCEnabled {
		// If no tokenizer path provided and in gRPC mode, try to fetch from worker
		logger.Info("No tokenizer path specified, attempting to fetch from worker",
			zap.Int("worker_count", len(workerRegistry.GetAll())),
		)

		// Create client pool for fetching tokenizer (if needed)
		// We'll need to create a temporary client pool or reuse the one created later
		// For now, create a temporary client pool
		tempClientPool := grpc.NewClientPool(logger)

		// Fetch tokenizer info from worker
		fetchedPath, fetchedChatTemplate, err := tokenizer.FetchTokenizerFromWorker(
			context.Background(),
			tempClientPool,
			workerRegistry,
			logger,
		)
		if err != nil {
			logger.Fatal("Failed to fetch tokenizer from worker",
				zap.Error(err),
				zap.String("hint", "Please provide --tokenizer-path or --model-path"),
			)
		}

		tokenizerPath = fetchedPath
		chatTemplatePath = fetchedChatTemplate

		logger.Info("Successfully fetched tokenizer info from worker",
			zap.String("tokenizer_path", tokenizerPath),
			zap.Bool("has_chat_template", chatTemplatePath != nil),
		)
	}

	if tokenizerPath != "" {
		var err error
		tok, err = tokenizer.CreateTokenizerWithChatTemplateBlocking(
			tokenizerPath,
			chatTemplatePath,
			logger,
		)
		if err != nil {
			logger.Fatal("Failed to create tokenizer",
				zap.String("path", tokenizerPath),
				zap.Error(err),
			)
		}
		logger.Info("Tokenizer loaded",
			zap.String("path", tokenizerPath),
			zap.Int("vocab_size", tok.GetVocabSize()),
		)
	} else {
		if cfg.GRPCEnabled {
			logger.Fatal("Tokenizer is required for gRPC mode, but could not be loaded or fetched from worker")
		}
		logger.Warn("No tokenizer configured - tokenization will use placeholder")
	}

	// Create router based on connection mode
	var r router.Router
	if cfg.ConnectionMode == core.ConnectionModeGRPC {
		// Create gRPC router
		grpcRouter, err := router.NewGrpcRouter(
			workerRegistry,
			policyRegistry,
			tok,
			nil, // TODO: Create tool parser factory
			nil, // TODO: Create reasoning parser factory
			logger,
		)
		if err != nil {
			logger.Fatal("Failed to create gRPC router", zap.Error(err))
		}
		r = grpcRouter
		logger.Info("Using gRPC router mode")
	} else {
		// Create HTTP router
		httpRouter, err := router.NewHttpRouter(
			workerRegistry,
			policyRegistry,
			logger,
		)
		if err != nil {
			logger.Fatal("Failed to create HTTP router", zap.Error(err))
		}
		r = httpRouter
		logger.Info("Using HTTP router mode")
	}

	// Start health checker
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	workerRegistry.StartHealthChecker(ctx, 60*time.Second)
	defer workerRegistry.StopHealthChecker()

	// Create registry adapter for HTTP server
	registryAdapter := core.NewRegistryAdapter(workerRegistry, logger)

	// Start HTTP server
	httpServer := server.NewHTTPServer(
		r,               // Use router interface (can be GrpcRouter or HttpRouter)
		registryAdapter, // Pass registry adapter for /workers endpoint
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
