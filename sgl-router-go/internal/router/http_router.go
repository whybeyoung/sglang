package router

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/sglang/sglang-router-go/internal/core"
	"github.com/sglang/sglang-router-go/internal/policy"
	"github.com/sglang/sglang-router-go/internal/protocols"
	"go.uber.org/zap"
)

// HttpRouter routes requests to HTTP workers
// Similar to Rust http::router::Router
type HttpRouter struct {
	workerRegistry *core.WorkerRegistry
	policyRegistry *policy.PolicyRegistry
	httpClient     *http.Client
	logger         *zap.Logger
}

// NewHttpRouter creates a new HTTP router
func NewHttpRouter(
	workerRegistry *core.WorkerRegistry,
	policyRegistry *policy.PolicyRegistry,
	logger *zap.Logger,
) (*HttpRouter, error) {
	// Create HTTP client with reasonable timeout
	httpClient := &http.Client{
		Timeout: 30 * time.Minute, // Allow for long-running requests
	}

	return &HttpRouter{
		workerRegistry: workerRegistry,
		policyRegistry: policyRegistry,
		httpClient:     httpClient,
		logger:         logger,
	}, nil
}

// RouteChat routes a chat completion request to HTTP workers
// Similar to Rust Router::route_chat
// Returns either *protocols.ChatCompletionResponse or *StreamingResponse
func (r *HttpRouter) RouteChat(
	ctx context.Context,
	request *protocols.ChatCompletionRequest,
	modelID *string,
) (interface{}, error) {
	// Select worker based on policy
	worker, err := r.selectWorker(modelID, request.Model)
	if err != nil {
		return nil, fmt.Errorf("no available worker: %w", err)
	}

	// Build worker URL
	workerURL := worker.URL()
	endpoint := "/v1/chat/completions"
	if workerURL[len(workerURL)-1] != '/' {
		endpoint = "/v1/chat/completions"
	}
	url := fmt.Sprintf("%s%s", workerURL, endpoint)

	// Serialize request
	reqBody, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Create HTTP request
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(reqBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")

	// Add API key if available
	if apiKey := worker.APIKey(); apiKey != nil && *apiKey != "" {
		httpReq.Header.Set("Authorization", fmt.Sprintf("Bearer %s", *apiKey))
	}

	// Send request
	resp, err := r.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("worker returned status %d: %s", resp.StatusCode, string(body))
	}

	// Parse response
	var chatResp protocols.ChatCompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&chatResp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &chatResp, nil
}

// RouteChatStream routes a streaming chat completion request
func (r *HttpRouter) RouteChatStream(
	ctx context.Context,
	request *protocols.ChatCompletionRequest,
	modelID *string,
	w http.ResponseWriter,
) error {
	// Select worker
	worker, err := r.selectWorker(modelID, request.Model)
	if err != nil {
		return fmt.Errorf("no available worker: %w", err)
	}

	// Build worker URL
	workerURL := worker.URL()
	url := fmt.Sprintf("%s/v1/chat/completions", workerURL)

	// Serialize request
	reqBody, err := json.Marshal(request)
	if err != nil {
		return fmt.Errorf("failed to marshal request: %w", err)
	}

	// Create HTTP request
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(reqBody))
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")

	// Add API key if available
	if apiKey := worker.APIKey(); apiKey != nil && *apiKey != "" {
		httpReq.Header.Set("Authorization", fmt.Sprintf("Bearer %s", *apiKey))
	}

	// Send request and proxy stream
	resp, err := r.httpClient.Do(httpReq)
	if err != nil {
		return fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	// Copy headers
	for k, v := range resp.Header {
		for _, val := range v {
			w.Header().Add(k, val)
		}
	}

	// Set status code
	w.WriteHeader(resp.StatusCode)

	// Stream response body
	_, err = io.Copy(w, resp.Body)
	return err
}

// RouteGenerate routes a generate request to HTTP workers
// Returns either *protocols.GenerateResponse or *StreamingResponse
func (r *HttpRouter) RouteGenerate(
	ctx context.Context,
	request *protocols.GenerateRequest,
	modelID *string,
) (interface{}, error) {
	// Select worker (GenerateRequest doesn't have Model field, use modelID)
	var requestModel string
	worker, err := r.selectWorker(modelID, requestModel)
	if err != nil {
		return nil, fmt.Errorf("no available worker: %w", err)
	}

	// Build worker URL
	workerURL := worker.URL()
	url := fmt.Sprintf("%s/generate", workerURL)

	// Serialize request
	reqBody, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Create HTTP request
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(reqBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")

	// Add API key if available
	if apiKey := worker.APIKey(); apiKey != nil && *apiKey != "" {
		httpReq.Header.Set("Authorization", fmt.Sprintf("Bearer %s", *apiKey))
	}

	// Send request
	resp, err := r.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("worker returned status %d: %s", resp.StatusCode, string(body))
	}

	// Parse response
	var genResp protocols.GenerateResponse
	if err := json.NewDecoder(resp.Body).Decode(&genResp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &genResp, nil
}

// RouteGenerateStream routes a streaming generate request
func (r *HttpRouter) RouteGenerateStream(
	ctx context.Context,
	request *protocols.GenerateRequest,
	modelID *string,
	w http.ResponseWriter,
) error {
	// Select worker (GenerateRequest doesn't have Model field, use modelID)
	var requestModel string
	worker, err := r.selectWorker(modelID, requestModel)
	if err != nil {
		return fmt.Errorf("no available worker: %w", err)
	}

	// Build worker URL
	workerURL := worker.URL()
	url := fmt.Sprintf("%s/generate", workerURL)

	// Serialize request
	reqBody, err := json.Marshal(request)
	if err != nil {
		return fmt.Errorf("failed to marshal request: %w", err)
	}

	// Create HTTP request
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(reqBody))
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")

	// Add API key if available
	if apiKey := worker.APIKey(); apiKey != nil && *apiKey != "" {
		httpReq.Header.Set("Authorization", fmt.Sprintf("Bearer %s", *apiKey))
	}

	// Send request and proxy stream
	resp, err := r.httpClient.Do(httpReq)
	if err != nil {
		return fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	// Copy headers
	for k, v := range resp.Header {
		for _, val := range v {
			w.Header().Add(k, val)
		}
	}

	// Set status code
	w.WriteHeader(resp.StatusCode)

	// Stream response body
	_, err = io.Copy(w, resp.Body)
	return err
}

// selectWorker selects a worker based on policy
func (r *HttpRouter) selectWorker(modelID *string, requestModel string) (core.Worker, error) {
	// Determine model ID
	var actualModelID string
	if modelID != nil && *modelID != "" {
		actualModelID = *modelID
	} else if requestModel != "" {
		actualModelID = requestModel
	} else {
		actualModelID = "default"
	}

	// Get all workers and filter by HTTP connection mode
	allWorkers := r.workerRegistry.GetAll()

	r.logger.Info("Selecting worker",
		zap.String("model_id", actualModelID),
		zap.Int("total_workers", len(allWorkers)),
	)

	// Filter healthy, available, HTTP-mode workers
	availableWorkers := make([]core.Worker, 0)
	for _, w := range allWorkers {
		connMode := w.ConnectionMode()
		isHealthy := w.IsHealthy()
		isAvailable := w.IsAvailable()

		r.logger.Info("Checking worker",
			zap.String("url", w.URL()),
			zap.String("connection_mode", string(connMode)),
			zap.Bool("is_healthy", isHealthy),
			zap.Bool("is_available", isAvailable),
			zap.String("model_id", w.ModelID()),
			zap.String("expected_mode", string(core.ConnectionModeHTTP)),
			zap.Bool("mode_match", connMode == core.ConnectionModeHTTP),
		)

		if connMode == core.ConnectionModeHTTP && isHealthy && isAvailable {
			// Also filter by model if specified
			if actualModelID == "default" || w.ModelID() == actualModelID {
				availableWorkers = append(availableWorkers, w)
				r.logger.Info("Worker added to available list",
					zap.String("url", w.URL()),
				)
			}
		}
	}

	if len(availableWorkers) == 0 {
		r.logger.Warn("No available HTTP workers",
			zap.String("model_id", actualModelID),
			zap.Int("total_workers", len(allWorkers)),
		)
		return nil, fmt.Errorf("no available HTTP workers")
	}

	// Get policy for this model
	p := r.policyRegistry.GetPolicyOrDefault(actualModelID)

	// Select worker using policy
	// Extract text for cache-aware policies (not available in HTTP router context)
	var textPtr *string
	workerIndex, ok := p.SelectWorker(availableWorkers, textPtr)
	if !ok || workerIndex < 0 || workerIndex >= len(availableWorkers) {
		return nil, fmt.Errorf("policy returned invalid worker index")
	}

	return availableWorkers[workerIndex], nil
}
