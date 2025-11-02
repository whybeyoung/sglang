package stages

// This package contains all pipeline stage implementations
// Similar to Rust src/routers/grpc/stages/mod.rs

// Stages:
// - PreparationStage: Tokenizes inputs, processes messages, filters tools
// - WorkerSelectionStage: Selects appropriate worker(s)
// - ClientAcquisitionStage: Acquires gRPC client connections
// - RequestBuildingStage: Builds gRPC request messages
// - DispatchMetadataStage: Prepares dispatch metadata
// - RequestExecutionStage: Executes requests via gRPC streams
// - ResponseProcessingStage: Processes and formats responses
