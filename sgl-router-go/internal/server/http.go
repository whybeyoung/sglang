package server

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/sglang/sglang-router-go/internal/protocols"
	"github.com/sglang/sglang-router-go/internal/router"
	"go.uber.org/zap"
)

// RegistryAccessor is an interface to access worker registry without import cycles
type RegistryAccessor interface {
	GetAll() []interface{}
	Stats() interface{}
	GetByModel(modelID string) []interface{}
	RegisterWorker(url, modelID, workerType, connectionMode string) error
}

// HTTPServer provides HTTP API endpoints
// Similar to Rust's HTTP router implementation
type HTTPServer struct {
	router         router.Router // Use interface instead of concrete type
	workerRegistry RegistryAccessor
	logger         *zap.Logger
	httpServer     *http.Server
}

// NewHTTPServer creates a new HTTP server
// Can accept either GrpcRouter or HttpRouter
func NewHTTPServer(
	r router.Router, // Accept Router interface
	workerRegistry RegistryAccessor,
	host string,
	port uint16,
	logger *zap.Logger,
) *HTTPServer {
	addr := fmt.Sprintf("%s:%d", host, port)

	mux := http.NewServeMux()
	srv := &HTTPServer{
		router:         r, // Use router interface
		workerRegistry: workerRegistry,
		logger:         logger,
		httpServer: &http.Server{
			Addr:    addr,
			Handler: mux,
		},
	}

	// Register routes
	srv.registerRoutes(mux)

	return srv
}

// registerRoutes registers all HTTP routes
func (s *HTTPServer) registerRoutes(mux *http.ServeMux) {
	// OpenAI-compatible endpoints
	mux.HandleFunc("/v1/chat/completions", s.handleChatCompletions)
	mux.HandleFunc("/generate", s.handleGenerate)

	// Health and management endpoints
	mux.HandleFunc("/health", s.handleHealth)
	mux.HandleFunc("/liveness", s.handleHealth)
	mux.HandleFunc("/readiness", s.handleReadiness)
	mux.HandleFunc("/workers", s.handleWorkers)
}

// handleChatCompletions handles POST /v1/chat/completions
func (s *HTTPServer) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req protocols.ChatCompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.logger.Warn("Failed to decode chat completion request", zap.Error(err))
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	// Extract model ID from request or query parameter
	modelID := req.GetModel()
	if modelID == "" {
		modelID = r.URL.Query().Get("model")
	}
	var modelIDPtr *string
	if modelID != "" {
		modelIDPtr = &modelID
	}

	// Route the request
	ctx, cancel := context.WithTimeout(r.Context(), 30*time.Minute)
	defer cancel()

	// Check if streaming request
	if req.Stream {
		err := s.router.RouteChatStream(ctx, &req, modelIDPtr, w)
		if err != nil {
			s.logger.Error("Failed to route streaming chat completion", zap.Error(err))
			http.Error(w, fmt.Sprintf("Internal server error: %v", err), http.StatusInternalServerError)
		}
		return
	}

	// Non-streaming request
	response, err := s.router.RouteChat(ctx, &req, modelIDPtr)
	if err != nil {
		s.logger.Error("Failed to route chat completion", zap.Error(err))
		http.Error(w, fmt.Sprintf("Internal server error: %v", err), http.StatusInternalServerError)
		return
	}

	// Check if this is a streaming response
	if streamResp, ok := response.(*router.StreamingResponse); ok {
		// Handle streaming SSE response
		s.handleStreamingResponse(w, streamResp)
		return
	}

	// Non-streaming response
	chatResponse, ok := response.(*protocols.ChatCompletionResponse)
	if !ok {
		s.logger.Error("Unexpected response type", zap.String("type", fmt.Sprintf("%T", response)))
		http.Error(w, "Internal server error: unexpected response type", http.StatusInternalServerError)
		return
	}

	// Write JSON response
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(chatResponse); err != nil {
		s.logger.Error("Failed to encode response", zap.Error(err))
	}
}

// handleGenerate handles POST /generate
func (s *HTTPServer) handleGenerate(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req protocols.GenerateRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.logger.Warn("Failed to decode generate request", zap.Error(err))
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	// Extract model ID from query parameter
	modelID := r.URL.Query().Get("model")
	var modelIDPtr *string
	if modelID != "" {
		modelIDPtr = &modelID
	}

	// Route the request
	ctx, cancel := context.WithTimeout(r.Context(), 30*time.Minute)
	defer cancel()

	// Check if streaming request
	if req.Stream {
		err := s.router.RouteGenerateStream(ctx, &req, modelIDPtr, w)
		if err != nil {
			s.logger.Error("Failed to route streaming generate request", zap.Error(err))
			http.Error(w, fmt.Sprintf("Internal server error: %v", err), http.StatusInternalServerError)
		}
		return
	}

	// Non-streaming request
	response, err := s.router.RouteGenerate(ctx, &req, modelIDPtr)
	if err != nil {
		s.logger.Error("Failed to route generate request", zap.Error(err))
		http.Error(w, fmt.Sprintf("Internal server error: %v", err), http.StatusInternalServerError)
		return
	}

	// Check if this is a streaming response
	if streamResp, ok := response.(*router.StreamingResponse); ok {
		// Handle streaming SSE response
		s.handleStreamingResponse(w, streamResp)
		return
	}

	// Non-streaming response
	generateResponse, ok := response.(*protocols.GenerateResponse)
	if !ok {
		s.logger.Error("Unexpected response type", zap.String("type", fmt.Sprintf("%T", response)))
		http.Error(w, "Internal server error: unexpected response type", http.StatusInternalServerError)
		return
	}

	// Write JSON response
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(generateResponse); err != nil {
		s.logger.Error("Failed to encode response", zap.Error(err))
	}
}

// handleHealth handles GET /health and /liveness
func (s *HTTPServer) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{
		"status": "healthy",
	})
}

// handleReadiness handles GET /readiness
func (s *HTTPServer) handleReadiness(w http.ResponseWriter, r *http.Request) {
	// TODO: Check if router is ready (has workers, etc.)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{
		"status": "ready",
	})
}

// handleWorkers handles GET /workers and POST /workers
func (s *HTTPServer) handleWorkers(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodGet {
		// List workers
		if s.workerRegistry == nil {
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]interface{}{
				"workers": []interface{}{},
				"total":   0,
				"message": "Worker registry not available",
			})
			return
		}

		workers := s.workerRegistry.GetAll()
		stats := s.workerRegistry.Stats()

		// Workers are already serialized by RegistryAdapter.GetAll()
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"workers": workers,
			"total":   len(workers),
			"stats":   stats,
		})
	} else if r.Method == http.MethodPost {
		// Register new worker
		var req struct {
			URL     string            `json:"url"`
			ModelID string            `json:"model_id,omitempty"`
			Type    string            `json:"type,omitempty"` // "regular", "prefill", "decode"
			Mode    string            `json:"mode,omitempty"` // "http" or "grpc"
			Labels  map[string]string `json:"labels,omitempty"`
		}

		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "Invalid request body", http.StatusBadRequest)
			return
		}

		if req.URL == "" {
			http.Error(w, "URL is required", http.StatusBadRequest)
			return
		}

		// Determine connection mode
		connectionMode := "grpc"
		if req.Mode != "" {
			connectionMode = req.Mode
		} else if strings.HasPrefix(req.URL, "http://") || strings.HasPrefix(req.URL, "https://") {
			connectionMode = "http"
		}

		// Determine worker type
		workerType := "regular"
		if req.Type != "" {
			workerType = req.Type
		}

		// Register worker
		if err := s.workerRegistry.RegisterWorker(req.URL, req.ModelID, workerType, connectionMode); err != nil {
			s.logger.Error("Failed to register worker",
				zap.String("url", req.URL),
				zap.Error(err),
			)
			http.Error(w, fmt.Sprintf("Failed to register worker: %v", err), http.StatusInternalServerError)
			return
		}

		s.logger.Info("Worker registered via API",
			zap.String("url", req.URL),
			zap.String("model_id", req.ModelID),
		)

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{
			"status":  "registered",
			"message": "Worker registered successfully",
		})
	} else {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
	}
}

// handleStreamingResponse writes a streaming SSE response
// Similar to Rust's SSE response handling
func (s *HTTPServer) handleStreamingResponse(w http.ResponseWriter, streamResp *router.StreamingResponse) {
	// Set SSE headers
	w.Header().Set("Content-Type", streamResp.ContentType)
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	// Set additional headers
	for k, v := range streamResp.Headers {
		w.Header().Set(k, v)
	}

	// Flush headers (important for SSE)
	if flusher, ok := w.(http.Flusher); ok {
		flusher.Flush()
	}

	// Copy stream data to response
	_, err := io.Copy(w, streamResp.Reader)
	if err != nil {
		s.logger.Error("Error writing streaming response", zap.Error(err))
	}
}

// Start starts the HTTP server
func (s *HTTPServer) Start() error {
	s.logger.Info("Starting HTTP server", zap.String("addr", s.httpServer.Addr))
	return s.httpServer.ListenAndServe()
}

// Shutdown gracefully shuts down the HTTP server
func (s *HTTPServer) Shutdown(ctx context.Context) error {
	s.logger.Info("Shutting down HTTP server")
	return s.httpServer.Shutdown(ctx)
}
