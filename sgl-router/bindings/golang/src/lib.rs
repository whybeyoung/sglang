//! FFI module for exposing sgl-router preprocessing and postprocessing functions
//! to C-compatible languages (e.g., Golang via cgo)
//!
//! This module provides C-compatible function signatures for:
//! - Tokenizer operations (encode, decode, chat template)
//! - Tool parser operations (parse tool calls)
//! - Tool constraint generation
//! - gRPC client SDK (complete request-response flow)
//!
//! # Safety
//! All functions marked with `#[no_mangle]` and `extern "C"` must be called
//! with valid pointers and follow the documented memory management rules.

// Re-export error types
pub use error::{SglErrorCode, set_error_message, set_error_message_fmt, clear_error_message};

// Re-export memory management functions
pub use memory::{sgl_free_string, sgl_free_token_ids};

// Re-export tokenizer functions
pub use tokenizer::{
    TokenizerHandle,
    sgl_tokenizer_create_from_file,
    sgl_tokenizer_encode,
    sgl_tokenizer_apply_chat_template,
    sgl_tokenizer_apply_chat_template_with_tools,
    sgl_tokenizer_decode,
    sgl_tokenizer_free,
};

// Re-export tool parser functions
pub use tool_parser::{
    ToolParserHandle,
    sgl_tool_parser_create,
    sgl_tool_parser_parse_complete,
    sgl_tool_parser_parse_incremental,
    sgl_tool_parser_reset,
    sgl_tool_parser_free,
};

// Re-export gRPC converter functions
pub use grpc_converter::{
    GrpcResponseConverterHandle,
    sgl_grpc_response_converter_create,
    sgl_grpc_response_converter_convert_chunk,
    sgl_grpc_response_converter_free,
};

// Re-export client SDK functions
pub use client::{
    SglangClientHandle,
    sgl_client_create,
    sgl_client_free,
};

// Re-export stream functions
pub use stream::{
    SglangStreamHandle,
    sgl_stream_read_next,
    sgl_stream_free,
};

// Re-export client stream function (defined in client.rs but used by stream)
pub use client::sgl_client_chat_completion_stream;

// Re-export utility functions
pub use utils::sgl_generate_tool_constraints;

// Re-export preprocessor functions
pub use preprocessor::{
    sgl_preprocess_chat_request,
    sgl_preprocess_chat_request_with_tokenizer,
    sgl_preprocessed_request_free,
};

// Re-export postprocessor functions
pub use postprocessor::{
    sgl_postprocess_stream_chunk,
    sgl_postprocess_stream_chunks_batch,
};

// Sub-modules
mod error;
mod memory;
mod tokenizer;
mod tool_parser;
mod grpc_converter;
mod client;
mod stream;
mod utils;
mod preprocessor;
mod postprocessor;

// Force linking of preprocessor and postprocessor functions
// This prevents LTO from removing unused FFI functions
// We create static references to the function pointers to ensure they are linked
#[used]
static PREPROCESSOR_PREPROCESS: unsafe extern "C" fn(
    *const std::os::raw::c_char,
    *const std::os::raw::c_char,
    *mut *mut std::os::raw::c_char,
    *mut *mut std::os::raw::c_uint,
    *mut usize,
    *mut *mut std::os::raw::c_char,
    *mut std::os::raw::c_int,
    *mut *mut std::os::raw::c_char,
) -> SglErrorCode = preprocessor::sgl_preprocess_chat_request;

#[used]
static PREPROCESSOR_FREE: unsafe extern "C" fn(
    *mut std::os::raw::c_char,
    *mut std::os::raw::c_uint,
    usize,
    *mut std::os::raw::c_char,
) = preprocessor::sgl_preprocessed_request_free;

#[used]
static PREPROCESSOR_WITH_TOKENIZER: unsafe extern "C" fn(
    *const std::os::raw::c_char,
    *mut tokenizer::TokenizerHandle,
    *mut *mut std::os::raw::c_char,
    *mut *mut std::os::raw::c_uint,
    *mut usize,
    *mut *mut std::os::raw::c_char,
    *mut std::os::raw::c_int,
    *mut *mut std::os::raw::c_char,
) -> error::SglErrorCode = preprocessor::sgl_preprocess_chat_request_with_tokenizer;

#[used]
static POSTPROCESSOR_CHUNK: unsafe extern "C" fn(
    *mut grpc_converter::GrpcResponseConverterHandle,
    *const std::os::raw::c_char,
    *mut *mut std::os::raw::c_char,
    *mut std::os::raw::c_int,
    *mut *mut std::os::raw::c_char,
) -> SglErrorCode = postprocessor::sgl_postprocess_stream_chunk;

#[used]
static POSTPROCESSOR_BATCH: unsafe extern "C" fn(
    *mut grpc_converter::GrpcResponseConverterHandle,
    *const std::os::raw::c_char,
    std::os::raw::c_int,
    *mut *mut std::os::raw::c_char,
    *mut std::os::raw::c_int,
    *mut *mut std::os::raw::c_char,
) -> SglErrorCode = postprocessor::sgl_postprocess_stream_chunks_batch;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_codes() {
        assert_eq!(SglErrorCode::Success as i32, 0);
        assert_eq!(SglErrorCode::InvalidArgument as i32, 1);
    }

    // Force compilation of preprocessor and postprocessor functions
    // This prevents LTO from removing unused FFI functions
    #[test]
    #[allow(dead_code)]
    fn test_ffi_functions_exist() {
        // These are just references to ensure functions are compiled
        // They won't actually be called, but their existence prevents LTO removal
        let _preprocess: unsafe extern "C" fn(*const std::os::raw::c_char, *const std::os::raw::c_char, *mut *mut std::os::raw::c_char, *mut *mut std::os::raw::c_uint, *mut usize, *mut *mut std::os::raw::c_char, *mut std::os::raw::c_int, *mut *mut std::os::raw::c_char) -> SglErrorCode = preprocessor::sgl_preprocess_chat_request;
        let _preprocess_with_tokenizer: unsafe extern "C" fn(*const std::os::raw::c_char, *mut tokenizer::TokenizerHandle, *mut *mut std::os::raw::c_char, *mut *mut std::os::raw::c_uint, *mut usize, *mut *mut std::os::raw::c_char, *mut std::os::raw::c_int, *mut *mut std::os::raw::c_char) -> SglErrorCode = preprocessor::sgl_preprocess_chat_request_with_tokenizer;
        let _preprocess_free: unsafe extern "C" fn(*mut std::os::raw::c_char, *mut std::os::raw::c_uint, usize, *mut std::os::raw::c_char) = preprocessor::sgl_preprocessed_request_free;
        let _postprocess: unsafe extern "C" fn(*mut grpc_converter::GrpcResponseConverterHandle, *const std::os::raw::c_char, *mut *mut std::os::raw::c_char, *mut std::os::raw::c_int, *mut *mut std::os::raw::c_char) -> SglErrorCode = postprocessor::sgl_postprocess_stream_chunk;
        let _postprocess_batch: unsafe extern "C" fn(*mut grpc_converter::GrpcResponseConverterHandle, *const std::os::raw::c_char, std::os::raw::c_int, *mut *mut std::os::raw::c_char, *mut std::os::raw::c_int, *mut *mut std::os::raw::c_char) -> SglErrorCode = postprocessor::sgl_postprocess_stream_chunks_batch;
    }
}
