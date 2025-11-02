package tokenizer

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/sglang/sglang-router-go/internal/core"
	"github.com/sglang/sglang-router-go/internal/grpc"
	"go.uber.org/zap"
)

// FetchTokenizerFromWorker fetches tokenizer and chat template from a worker via gRPC
// Returns the local path where tokenizer files are cached (or temp path for in-memory content)
func FetchTokenizerFromWorker(
	ctx context.Context,
	clientPool *grpc.ClientPool,
	workerRegistry *core.WorkerRegistry,
	logger *zap.Logger,
) (string, *string, error) {
	// Find a gRPC worker to fetch from
	// Note: We don't require worker to be healthy, as health check might fail
	// but we can still try to connect and fetch tokenizer info
	workers := workerRegistry.GetAll()
	var selectedWorker core.Worker
	for _, worker := range workers {
		if worker.ConnectionMode() == core.ConnectionModeGRPC {
			selectedWorker = worker
			logger.Info("Selected gRPC worker for tokenizer fetch",
				zap.String("worker_url", worker.URL()),
				zap.String("model_id", worker.ModelID()),
				zap.Bool("healthy", worker.IsHealthy()),
			)
			break
		}
	}

	if selectedWorker == nil {
		return "", nil, fmt.Errorf("no gRPC worker available to fetch tokenizer from")
	}

	logger.Info("Fetching tokenizer from worker",
		zap.String("worker_url", selectedWorker.URL()),
		zap.String("model_id", selectedWorker.ModelID()),
	)

	// Create context with longer timeout for fetching tokenizer
	// This includes time for connection establishment and RPC call
	fetchCtx, fetchCancel := context.WithTimeout(ctx, 30*time.Second)
	defer fetchCancel()

	// Get gRPC client
	logger.Debug("Establishing gRPC connection to worker",
		zap.String("worker_url", selectedWorker.URL()),
	)
	clientWrapper, err := clientPool.GetClientWrapper(fetchCtx, selectedWorker)
	if err != nil {
		return "", nil, fmt.Errorf("failed to get gRPC client (connection may be slow or worker unreachable): %w", err)
	}

	// Try to fetch tokenizer files directly via GetTokenizerInfo RPC first
	// This allows fetching files even without shared filesystem
	logger.Info("Attempting to fetch tokenizer files directly via gRPC",
		zap.String("worker_url", selectedWorker.URL()),
	)

	// Request key tokenizer files
	requestedFiles := []string{
		"tokenizer.json",
		"tokenizer_config.json",
		"chat_template.jinja",
		"chat_template.json",
	}

	tokenizerInfo, err := clientWrapper.GetTokenizerInfo(fetchCtx, requestedFiles)
	if err == nil && tokenizerInfo != nil && tokenizerInfo.Success && len(tokenizerInfo.Files) > 0 {
		// Successfully fetched tokenizer files via gRPC
		logger.Info("Successfully fetched tokenizer files via gRPC",
			zap.Int("file_count", len(tokenizerInfo.Files)),
			zap.String("base_path", tokenizerInfo.BasePath),
		)

		// Create a temporary directory for minimal file writes
		// We need to write tokenizer.json at minimum because the tokenizer libraries require a file path
		tempDir, err := os.MkdirTemp("", "sglang-router-tokenizer-*")
		if err != nil {
			return "", nil, fmt.Errorf("failed to create temp directory: %w", err)
		}

		// Extract tokenizer.json (required)
		tokenizerJSON, ok := tokenizerInfo.Files["tokenizer.json"]
		if !ok || tokenizerJSON == "" {
			os.RemoveAll(tempDir)
			return "", nil, fmt.Errorf("tokenizer.json not found in fetched files")
		}

		// Write only the minimal required files (tokenizer.json is required by libraries)
		tokenizerPath := filepath.Join(tempDir, "tokenizer.json")
		if err := os.WriteFile(tokenizerPath, []byte(tokenizerJSON), 0644); err != nil {
			os.RemoveAll(tempDir)
			return "", nil, fmt.Errorf("failed to write tokenizer.json: %w", err)
		}

		// Write tokenizer_config.json if available (may contain chat template)
		if tokenizerConfig, ok := tokenizerInfo.Files["tokenizer_config.json"]; ok && tokenizerConfig != "" {
			configPath := filepath.Join(tempDir, "tokenizer_config.json")
			if err := os.WriteFile(configPath, []byte(tokenizerConfig), 0644); err != nil {
				logger.Warn("Failed to write tokenizer_config.json, continuing",
					zap.Error(err),
				)
			}
		}

		// Try to discover chat template from saved files
		// Priority: 1. Standalone chat_template files, 2. tokenizer_config.json
		var chatTemplate *string

		// First, check if standalone chat template files exist
		if chatTemplateContent, ok := tokenizerInfo.Files["chat_template.jinja"]; ok && chatTemplateContent != "" {
			templatePath := filepath.Join(tempDir, "chat_template.jinja")
			if err := os.WriteFile(templatePath, []byte(chatTemplateContent), 0644); err != nil {
				logger.Warn("Failed to write chat_template.jinja, continuing",
					zap.Error(err),
				)
			} else {
				chatTemplate = &templatePath
				logger.Info("Using standalone chat_template.jinja from worker",
					zap.String("source_file", "chat_template.jinja (from GetTokenizerInfo RPC)"),
					zap.String("saved_path", templatePath),
					zap.String("temp_dir", tempDir),
				)
			}
		} else if chatTemplateContent, ok := tokenizerInfo.Files["chat_template.json"]; ok && chatTemplateContent != "" {
			templatePath := filepath.Join(tempDir, "chat_template.json")
			if err := os.WriteFile(templatePath, []byte(chatTemplateContent), 0644); err != nil {
				logger.Warn("Failed to write chat_template.json, continuing",
					zap.Error(err),
				)
			} else {
				chatTemplate = &templatePath
				logger.Info("Using standalone chat_template.json from worker",
					zap.String("source_file", "chat_template.json (from GetTokenizerInfo RPC)"),
					zap.String("saved_path", templatePath),
					zap.String("temp_dir", tempDir),
				)
			}
		}

		// If no standalone chat template, try to extract from tokenizer_config.json
		if chatTemplate == nil {
			// Check if tokenizer_config.json exists and extract chat_template from it
			configPath := filepath.Join(tempDir, "tokenizer_config.json")
			if _, err := os.Stat(configPath); err == nil {
				templateContent, err := LoadChatTemplateFromConfig(configPath)
				if err == nil && templateContent != nil {
					// Write extracted template content to a temporary file
					// Since NewHuggingFaceTokenizerWithChatTemplate expects a file path
					templatePath := filepath.Join(tempDir, "chat_template.jinja")
					if err := os.WriteFile(templatePath, []byte(*templateContent), 0644); err != nil {
						logger.Warn("Failed to write extracted chat template to file",
							zap.Error(err),
						)
					} else {
						chatTemplate = &templatePath
						logger.Info("Extracted and saved chat template from tokenizer_config.json",
							zap.String("source_file", configPath),
							zap.String("saved_path", templatePath),
							zap.String("temp_dir", tempDir),
						)
					}
				}
			}
		}

		logger.Info("Tokenizer prepared from in-memory content (minimal disk write)",
			zap.String("tokenizer_path", tokenizerPath),
			zap.String("temp_dir", tempDir),
			zap.Bool("has_chat_template", chatTemplate != nil),
			zap.Int("files_count", len(tokenizerInfo.Files)),
		)

		return tokenizerPath, chatTemplate, nil
	}

	// Fallback: Try GetModelInfo and use shared filesystem
	if err != nil {
		logger.Warn("GetTokenizerInfo RPC failed, falling back to GetModelInfo",
			zap.Error(err),
		)
	} else if tokenizerInfo == nil || !tokenizerInfo.Success {
		logger.Warn("GetTokenizerInfo returned failure, falling back to GetModelInfo",
			zap.String("error", tokenizerInfo.ErrorMessage),
		)
	}

	logger.Info("Falling back to GetModelInfo and shared filesystem approach")

	// Get model info using proto types
	logger.Debug("Calling GetModelInfo RPC",
		zap.String("worker_url", selectedWorker.URL()),
	)
	modelInfo, err := clientWrapper.GetModelInfo(fetchCtx)
	if err != nil {
		return "", nil, fmt.Errorf("failed to get model info from worker: %w", err)
	}

	tokenizerPath := modelInfo.TokenizerPath
	if tokenizerPath == "" {
		// Try model_path as fallback
		if modelInfo.ModelPath != "" {
			tokenizerPath = modelInfo.ModelPath
			logger.Info("Using model_path as tokenizer path",
				zap.String("model_path", modelInfo.ModelPath),
			)
		} else {
			return "", nil, fmt.Errorf("worker returned empty tokenizer_path and model_path")
		}
	}

	logger.Info("Got tokenizer path from worker",
		zap.String("tokenizer_path", tokenizerPath),
		zap.String("model_path", modelInfo.ModelPath),
	)

	// Validate tokenizer path exists
	if _, err := os.Stat(tokenizerPath); os.IsNotExist(err) {
		return "", nil, fmt.Errorf("tokenizer path from worker does not exist locally: %s (consider using --tokenizer-path or --model-path)", tokenizerPath)
	}

	// Auto-discover chat template from tokenizer directory
	var chatTemplate *string
	tokenizerDir := tokenizerPath
	if info, err := os.Stat(tokenizerPath); err == nil && !info.IsDir() {
		// If it's a file, use its directory
		tokenizerDir = filepath.Dir(tokenizerPath)
	}

	// Try to discover chat template
	discovered := discoverChatTemplateInDir(tokenizerDir)
	if discovered != nil {
		chatTemplate = discovered
		logger.Info("Discovered chat template from worker tokenizer directory",
			zap.String("chat_template", *discovered),
		)
	}

	return tokenizerPath, chatTemplate, nil
}
