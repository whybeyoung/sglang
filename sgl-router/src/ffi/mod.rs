//! FFI module for exposing sgl-router preprocessing and postprocessing functions
//! to C-compatible languages (e.g., Golang via cgo)
//!
//! This module provides C-compatible function signatures for:
//! - Tokenizer operations (encode, decode, chat template)
//! - Tool parser operations (parse tool calls)
//! - Tool constraint generation
//! - JSON schema parsing
//!
//! # Safety
//! All functions marked with `#[no_mangle]` and `extern "C"` must be called
//! with valid pointers and follow the documented memory management rules.

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int};
use std::ptr;

use std::sync::Arc;
use serde_json::{self, json, Value};
use tokio::runtime::Runtime;
use once_cell::sync::Lazy;
use crate::tokenizer::{
    create_tokenizer_from_file,
    traits::Tokenizer as TokenizerTrait,
    chat_template::ChatTemplateParams,
    huggingface::HuggingFaceTokenizer,
};
use crate::tokenizer::traits::Tokenizer;
use crate::tool_parser::{ParserFactory, ToolParser};
use crate::protocols::common::{
    Tool, ToolChoice, ToolChoiceValue, ToolCallDelta, FunctionCallDelta, Usage, StringOrArray,
};
use crate::tokenizer::stop::StopSequenceDecoder;
use crate::grpc_client::{proto, sglang_scheduler::{SglangSchedulerClient, AbortOnDropStream}};
use crate::protocols::chat::ChatCompletionRequest;
use crate::routers::grpc::{utils::{process_chat_messages, generate_tool_constraints}, ProcessedMessages};
use std::collections::HashMap;
use futures_util::StreamExt;
use uuid::Uuid;

// ============================================================================
// Error Handling
// ============================================================================

/// Error codes returned by FFI functions
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SglErrorCode {
    Success = 0,
    InvalidArgument = 1,
    TokenizationError = 2,
    ParsingError = 3,
    MemoryError = 4,
    UnknownError = 99,
}

/// Helper to convert Rust Result to error code and optional error message
fn handle_result<T>(
    result: Result<T, String>,
    error_out: *mut *mut c_char,
) -> Result<T, SglErrorCode> {
    match result {
        Ok(value) => {
            if !error_out.is_null() {
                unsafe {
                    *error_out = ptr::null_mut();
                }
            }
            Ok(value)
        }
        Err(e) => {
            if !error_out.is_null() {
                unsafe {
                    *error_out = CString::new(e.clone())
                        .ok()
                        .map(|s| s.into_raw())
                        .unwrap_or(ptr::null_mut());
                }
            }
            Err(SglErrorCode::UnknownError)
        }
    }
}

// ============================================================================
// Memory Management
// ============================================================================

/// Free a C string allocated by Rust
///
/// # Safety
/// This function must only be called with pointers returned by other FFI functions.
/// Calling with arbitrary pointers or multiple times on the same pointer is undefined behavior.
#[no_mangle]
pub unsafe extern "C" fn sgl_free_string(s: *mut c_char) {
    if !s.is_null() {
        let _ = CString::from_raw(s);
    }
}

/// Free a token ID array allocated by sgl_tokenizer_encode
///
/// # Safety
/// This function must only be called with pointers returned by sgl_tokenizer_encode.
/// Calling with arbitrary pointers or multiple times on the same pointer is undefined behavior.
#[no_mangle]
pub unsafe extern "C" fn sgl_free_token_ids(ptr: *mut u32, count: usize) {
    if !ptr.is_null() && count > 0 {
        let _ = Vec::from_raw_parts(ptr, count, count);
    }
}

// ============================================================================
// Tokenizer FFI
// ============================================================================

/// Opaque handle for a tokenizer instance
#[repr(C)]
pub struct TokenizerHandle {
    tokenizer: Arc<dyn TokenizerTrait>,
}

/// Create a tokenizer from a file path
///
/// # Arguments
/// * `path` - Path to tokenizer.json file (null-terminated C string)
/// * `error_out` - Optional pointer to receive error message (must be freed with sgl_free_string)
///
/// # Returns
/// * Pointer to TokenizerHandle on success, null on failure
///
/// # Safety
/// The returned handle must be freed with `sgl_tokenizer_free`.
#[no_mangle]
pub unsafe extern "C" fn sgl_tokenizer_create_from_file(
    path: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut TokenizerHandle {
    if path.is_null() {
        if !error_out.is_null() {
            let msg = CString::new("path cannot be null").unwrap();
            *error_out = msg.into_raw();
        }
        return ptr::null_mut();
    }

    let path_str = match CStr::from_ptr(path).to_str() {
        Ok(s) => s,
        Err(e) => {
            if !error_out.is_null() {
                let msg = CString::new(format!("Invalid UTF-8 in path: {}", e)).unwrap();
                *error_out = msg.into_raw();
            }
            return ptr::null_mut();
        }
    };

    match create_tokenizer_from_file(path_str) {
        Ok(tokenizer) => {
            if !error_out.is_null() {
                *error_out = ptr::null_mut();
            }
            Box::into_raw(Box::new(TokenizerHandle {
                tokenizer,
            }))
        }
        Err(e) => {
            if !error_out.is_null() {
                let msg = CString::new(e.to_string()).unwrap();
                *error_out = msg.into_raw();
            }
            ptr::null_mut()
        }
    }
}

/// Encode text to token IDs
///
/// # Arguments
/// * `handle` - Tokenizer handle (must not be null)
/// * `text` - Input text (null-terminated C string)
/// * `token_ids_out` - Pointer to receive array of token IDs (must be freed with free())
/// * `token_count_out` - Pointer to receive token count
/// * `error_out` - Optional pointer to receive error message
///
/// # Returns
/// * SglErrorCode::Success on success, error code on failure
///
/// # Safety
/// The token_ids_out array must be freed with free() after use.
#[no_mangle]
pub unsafe extern "C" fn sgl_tokenizer_encode(
    handle: *mut TokenizerHandle,
    text: *const c_char,
    token_ids_out: *mut *mut u32,
    token_count_out: *mut usize,
    error_out: *mut *mut c_char,
) -> SglErrorCode {
    if handle.is_null() || text.is_null() || token_ids_out.is_null() || token_count_out.is_null() {
        if !error_out.is_null() {
            let msg = CString::new("Invalid arguments: null pointer").unwrap();
            *error_out = msg.into_raw();
        }
        return SglErrorCode::InvalidArgument;
    }

    let text_str = match CStr::from_ptr(text).to_str() {
        Ok(s) => s,
        Err(_) => {
            if !error_out.is_null() {
                let msg = CString::new("Invalid UTF-8 in text").unwrap();
                *error_out = msg.into_raw();
            }
            return SglErrorCode::InvalidArgument;
        }
    };

    let tokenizer = &(*handle).tokenizer;
    match tokenizer.encode(text_str) {
        Ok(encoding) => {
            let token_ids = encoding.token_ids();
            let count = token_ids.len();

            // Allocate memory for token IDs using Vec, then leak to give ownership to C
            let vec = token_ids.to_vec();
            let ptr = vec.as_ptr() as *mut u32;
            let _ = std::mem::ManuallyDrop::new(vec);

            *token_ids_out = ptr;
            *token_count_out = count;

            if !error_out.is_null() {
                *error_out = ptr::null_mut();
            }

            SglErrorCode::Success
        }
        Err(e) => {
            if !error_out.is_null() {
                let msg = CString::new(e.to_string()).unwrap();
                *error_out = msg.into_raw();
            }
            SglErrorCode::TokenizationError
        }
    }
}

/// Apply chat template to messages with tools support
///
/// # Arguments
/// * `handle` - Tokenizer handle
/// * `messages_json` - JSON string of messages array
/// * `tools_json` - Optional JSON string of tools array (null or empty string for no tools)
/// * `result_out` - Pointer to receive result string (must be freed with sgl_free_string)
/// * `error_out` - Optional pointer to receive error message
///
/// # Returns
/// * SglErrorCode::Success on success, error code on failure
#[no_mangle]
pub unsafe extern "C" fn sgl_tokenizer_apply_chat_template_with_tools(
    handle: *mut TokenizerHandle,
    messages_json: *const c_char,
    tools_json: *const c_char,
    result_out: *mut *mut c_char,
    error_out: *mut *mut c_char,
) -> SglErrorCode {
    if handle.is_null() || messages_json.is_null() || result_out.is_null() {
        if !error_out.is_null() {
            let msg = CString::new("Invalid arguments: null pointer").unwrap();
            *error_out = msg.into_raw();
        }
        return SglErrorCode::InvalidArgument;
    }

    let messages_str = match CStr::from_ptr(messages_json).to_str() {
        Ok(s) => s,
        Err(_) => {
            if !error_out.is_null() {
                let msg = CString::new("Invalid UTF-8 in messages_json").unwrap();
                *error_out = msg.into_raw();
            }
            return SglErrorCode::InvalidArgument;
        }
    };

    // Parse JSON messages
    let messages: Vec<Value> = match serde_json::from_str(messages_str) {
        Ok(msgs) => msgs,
        Err(e) => {
            if !error_out.is_null() {
                let msg = CString::new(format!("Failed to parse messages JSON: {}", e)).unwrap();
                *error_out = msg.into_raw();
            }
            return SglErrorCode::InvalidArgument;
        }
    };

    // Parse tools JSON if provided
    let tools: Option<Vec<Value>> = if tools_json.is_null() {
        None
    } else {
        let tools_str = match CStr::from_ptr(tools_json).to_str() {
            Ok(s) => {
                if s.is_empty() {
                    None
                } else {
                    match serde_json::from_str::<Vec<Value>>(s) {
                        Ok(t) => Some(t),
                        Err(e) => {
                            if !error_out.is_null() {
                                let msg = CString::new(format!("Failed to parse tools JSON: {}", e)).unwrap();
                                *error_out = msg.into_raw();
                            }
                            return SglErrorCode::InvalidArgument;
                        }
                    }
                }
            }
            Err(_) => {
                if !error_out.is_null() {
                    let msg = CString::new("Invalid UTF-8 in tools_json").unwrap();
                    *error_out = msg.into_raw();
                }
                return SglErrorCode::InvalidArgument;
            }
        };
        tools_str
    };

    // Get the tokenizer from handle
    let handle_ref = &*handle;
    let tokenizer = &handle_ref.tokenizer;

    // Try to downcast to HuggingFaceTokenizer
    if let Some(hf_tokenizer) = tokenizer.as_any().downcast_ref::<HuggingFaceTokenizer>() {
        // Apply chat template with tools
        let empty_docs: [Value; 0] = [];
        let tools_slice = tools.as_ref().map(|t| t.as_slice());
        let params = ChatTemplateParams {
            add_generation_prompt: true,
            tools: tools_slice,
            documents: Some(&empty_docs),
            template_kwargs: None,
        };

        match hf_tokenizer.apply_chat_template(&messages, params) {
            Ok(result) => {
                let result_cstr = match CString::new(result) {
                    Ok(s) => s,
                    Err(e) => {
                        if !error_out.is_null() {
                            let msg = CString::new(format!("Failed to create result string: {}", e)).unwrap();
                            *error_out = msg.into_raw();
                        }
                        return SglErrorCode::MemoryError;
                    }
                };
                *result_out = result_cstr.into_raw();
                if !error_out.is_null() {
                    *error_out = ptr::null_mut();
                }
                return SglErrorCode::Success;
            }
            Err(e) => {
                if !error_out.is_null() {
                    let msg = CString::new(format!("Failed to apply chat template: {}", e)).unwrap();
                    *error_out = msg.into_raw();
                }
                return SglErrorCode::TokenizationError;
            }
        }
    } else {
        // Tokenizer doesn't support chat templates
        if !error_out.is_null() {
            let msg = CString::new("Chat template is only supported for HuggingFace tokenizers").unwrap();
            *error_out = msg.into_raw();
        }
        return SglErrorCode::TokenizationError;
    }
}

/// Apply chat template to messages
///
/// # Arguments
/// * `handle` - Tokenizer handle
/// * `messages_json` - JSON string of messages array
/// * `result_out` - Pointer to receive result string (must be freed with sgl_free_string)
/// * `error_out` - Optional pointer to receive error message
///
/// # Returns
/// * SglErrorCode::Success on success, error code on failure
#[no_mangle]
pub unsafe extern "C" fn sgl_tokenizer_apply_chat_template(
    handle: *mut TokenizerHandle,
    messages_json: *const c_char,
    result_out: *mut *mut c_char,
    error_out: *mut *mut c_char,
) -> SglErrorCode {
    if handle.is_null() || messages_json.is_null() || result_out.is_null() {
        if !error_out.is_null() {
            let msg = CString::new("Invalid arguments: null pointer").unwrap();
            *error_out = msg.into_raw();
        }
        return SglErrorCode::InvalidArgument;
    }

    let messages_str = match CStr::from_ptr(messages_json).to_str() {
        Ok(s) => s,
        Err(_) => {
            if !error_out.is_null() {
                let msg = CString::new("Invalid UTF-8 in messages_json").unwrap();
                *error_out = msg.into_raw();
            }
            return SglErrorCode::InvalidArgument;
        }
    };

    // Parse JSON messages
    let messages: Vec<Value> = match serde_json::from_str(messages_str) {
        Ok(msgs) => msgs,
        Err(e) => {
            if !error_out.is_null() {
                let msg = CString::new(format!("Failed to parse messages JSON: {}", e)).unwrap();
                *error_out = msg.into_raw();
            }
            return SglErrorCode::InvalidArgument;
        }
    };

    // Get the tokenizer from handle
    let handle_ref = &*handle;
    let tokenizer = &handle_ref.tokenizer;

    // Try to downcast to HuggingFaceTokenizer
    if let Some(hf_tokenizer) = tokenizer.as_any().downcast_ref::<HuggingFaceTokenizer>() {
        // Apply chat template with default parameters
        // Use empty arrays instead of None to avoid template errors
        // Set add_generation_prompt to true so the model knows to start generating
        let empty_tools: [Value; 0] = [];
        let empty_docs: [Value; 0] = [];
        let params = ChatTemplateParams {
            add_generation_prompt: true,  // Important: tells the model to start generating
            tools: Some(&empty_tools),
            documents: Some(&empty_docs),
            template_kwargs: None,
        };

        match hf_tokenizer.apply_chat_template(&messages, params) {
            Ok(result) => {
                let result_cstr = match CString::new(result) {
                    Ok(s) => s,
                    Err(e) => {
                        if !error_out.is_null() {
                            let msg = CString::new(format!("Failed to create result string: {}", e)).unwrap();
                            *error_out = msg.into_raw();
                        }
                        return SglErrorCode::MemoryError;
                    }
                };
                *result_out = result_cstr.into_raw();
                if !error_out.is_null() {
                    *error_out = ptr::null_mut();
                }
                return SglErrorCode::Success;
            }
            Err(e) => {
                if !error_out.is_null() {
                    let msg = CString::new(format!("Failed to apply chat template: {}", e)).unwrap();
                    *error_out = msg.into_raw();
                }
                return SglErrorCode::TokenizationError;
            }
        }
    } else {
        // Tokenizer doesn't support chat templates
        if !error_out.is_null() {
            let msg = CString::new("Chat template is only supported for HuggingFace tokenizers").unwrap();
            *error_out = msg.into_raw();
        }
        return SglErrorCode::TokenizationError;
    }
}

/// Decode token IDs to text
///
/// # Arguments
/// * `handle` - Tokenizer handle
/// * `token_ids` - Array of token IDs
/// * `token_count` - Number of tokens
/// * `skip_special_tokens` - Whether to skip special tokens
/// * `result_out` - Pointer to receive result string (must be freed with sgl_free_string)
/// * `error_out` - Optional pointer to receive error message
///
/// # Returns
/// * SglErrorCode::Success on success, error code on failure
#[no_mangle]
pub unsafe extern "C" fn sgl_tokenizer_decode(
    handle: *mut TokenizerHandle,
    token_ids: *const u32,
    token_count: usize,
    skip_special_tokens: c_int,
    result_out: *mut *mut c_char,
    error_out: *mut *mut c_char,
) -> SglErrorCode {
    if handle.is_null() || token_ids.is_null() || result_out.is_null() {
        if !error_out.is_null() {
            let msg = CString::new("Invalid arguments: null pointer").unwrap();
            *error_out = msg.into_raw();
        }
        return SglErrorCode::InvalidArgument;
    }

    if token_count == 0 {
        let empty = CString::new("").unwrap();
        *result_out = empty.into_raw();
        if !error_out.is_null() {
            *error_out = ptr::null_mut();
        }
        return SglErrorCode::Success;
    }

    // Convert C array to Rust slice
    let token_slice = std::slice::from_raw_parts(token_ids, token_count);

    let tokenizer = &(*handle).tokenizer;
    match tokenizer.decode(token_slice, skip_special_tokens != 0) {
        Ok(text) => {
            let result_cstr = match CString::new(text) {
                Ok(s) => s,
                Err(e) => {
                    if !error_out.is_null() {
                        let msg = CString::new(format!("Failed to create result string: {}", e)).unwrap();
                        *error_out = msg.into_raw();
                    }
                    return SglErrorCode::MemoryError;
                }
            };
            *result_out = result_cstr.into_raw();
            if !error_out.is_null() {
                *error_out = ptr::null_mut();
            }
            SglErrorCode::Success
        }
        Err(e) => {
            if !error_out.is_null() {
                let msg = CString::new(e.to_string()).unwrap();
                *error_out = msg.into_raw();
            }
            SglErrorCode::TokenizationError
        }
    }
}

/// Free a tokenizer handle
///
/// # Safety
/// This function must only be called once per handle, and the handle must not be used after calling.
#[no_mangle]
pub unsafe extern "C" fn sgl_tokenizer_free(handle: *mut TokenizerHandle) {
    if !handle.is_null() {
        let _ = Box::from_raw(handle);
    }
}

// ============================================================================
// Tool Parser FFI
// ============================================================================

/// Global parser factory (initialized once)
static PARSER_FACTORY: Lazy<ParserFactory> = Lazy::new(|| ParserFactory::new());

/// Global tokio runtime for async operations
static RUNTIME: Lazy<Runtime> = Lazy::new(|| {
    Runtime::new().expect("Failed to create tokio runtime for tool parser FFI")
});

/// Opaque handle for a tool parser instance
/// Note: For streaming, we need mutable access, so we use Arc<Mutex<>> internally
/// Note: This is an opaque handle, C code doesn't access fields directly
pub struct ToolParserHandle {
    parser: Arc<tokio::sync::Mutex<Box<dyn ToolParser>>>,
    model: String, // Store model name for ID generation
    history_tool_calls_count: usize, // Track tool call count for ID generation
    tool_index_to_id: HashMap<usize, String>, // Map tool_index to ID for incremental updates
}

/// Create a tool parser
///
/// # Arguments
/// * `parser_type` - Parser type name (e.g., "json", "llama", "mistral") or model name (e.g., "gpt-4")
/// * `error_out` - Optional pointer to receive error message
///
/// # Returns
/// * Pointer to ToolParserHandle on success, null on failure
#[no_mangle]
pub unsafe extern "C" fn sgl_tool_parser_create(
    parser_type: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut ToolParserHandle {
    if parser_type.is_null() {
        if !error_out.is_null() {
            let msg = CString::new("parser_type cannot be null").unwrap();
            *error_out = msg.into_raw();
        }
        return ptr::null_mut();
    }

    let type_str = match CStr::from_ptr(parser_type).to_str() {
        Ok(s) => s,
        Err(_) => {
            if !error_out.is_null() {
                let msg = CString::new("Invalid UTF-8 in parser_type").unwrap();
                *error_out = msg.into_raw();
            }
            return ptr::null_mut();
        }
    };

    // Create parser using factory
    // The factory will determine the parser type based on model name or use the provided type
    let parser = if let Some(parser_box) = PARSER_FACTORY.registry().create_for_model(type_str) {
        parser_box
    } else if let Some(parser_box) = PARSER_FACTORY.registry().create_parser(type_str) {
        parser_box
    } else {
        if !error_out.is_null() {
            let msg = CString::new(format!("Unknown parser type: {}", type_str)).unwrap();
            *error_out = msg.into_raw();
        }
        return ptr::null_mut();
    };

    Box::into_raw(Box::new(ToolParserHandle {
        parser: Arc::new(tokio::sync::Mutex::new(parser)),
        model: type_str.to_string(),
        history_tool_calls_count: 0,
        tool_index_to_id: HashMap::new(),
    }))
}

/// Parse complete tool calls from text
///
/// # Arguments
/// * `handle` - Tool parser handle
/// * `text` - Input text to parse
/// * `result_json_out` - Pointer to receive JSON result (must be freed with sgl_free_string)
/// * `error_out` - Optional pointer to receive error message
///
/// # Returns
/// * SglErrorCode::Success on success, error code on failure
///
/// # Result JSON Format
/// ```json
/// {
///   "normal_text": "...",
///   "tool_calls": [
///     {
///       "id": "...",
///       "type": "function",
///       "function": {
///         "name": "...",
///         "arguments": "..."
///       }
///     }
///   ]
/// }
/// ```
#[no_mangle]
pub unsafe extern "C" fn sgl_tool_parser_parse_complete(
    handle: *mut ToolParserHandle,
    text: *const c_char,
    result_json_out: *mut *mut c_char,
    error_out: *mut *mut c_char,
) -> SglErrorCode {
    if handle.is_null() || text.is_null() || result_json_out.is_null() {
        if !error_out.is_null() {
            let msg = CString::new("Invalid arguments: null pointer").unwrap();
            *error_out = msg.into_raw();
        }
        return SglErrorCode::InvalidArgument;
    }

    let text_str = match CStr::from_ptr(text).to_str() {
        Ok(s) => s,
        Err(_) => {
            if !error_out.is_null() {
                let msg = CString::new("Invalid UTF-8 in text").unwrap();
                *error_out = msg.into_raw();
            }
            return SglErrorCode::InvalidArgument;
        }
    };

    let handle_ref = &*handle;
    let parser = Arc::clone(&handle_ref.parser);
    let model = handle_ref.model.clone();
    let history_count = handle_ref.history_tool_calls_count;

    // Use tokio runtime to run async code
    let result = RUNTIME.block_on(async {
        let parser_guard = parser.lock().await;
        parser_guard.parse_complete(text_str).await
    });

    match result {
        Ok((normal_text, tool_calls)) => {
            // Convert Rust ToolCall to OpenAI format
            let openai_tool_calls: Vec<Value> = tool_calls
                .into_iter()
                .enumerate()
                .map(|(index, tc)| {
                    // Generate ID for this tool call
                    let id = generate_tool_call_id(&model, &tc.function.name, index, history_count);
                    json!({
                        "id": id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    })
                })
                .collect();

            // Build result JSON
            let result_json = json!({
                "normal_text": normal_text,
                "tool_calls": openai_tool_calls
            });

            let result_str = match serde_json::to_string(&result_json) {
                Ok(s) => s,
                Err(e) => {
                    if !error_out.is_null() {
                        let msg = CString::new(format!("Failed to serialize JSON: {}", e)).unwrap();
                        *error_out = msg.into_raw();
                    }
                    return SglErrorCode::ParsingError;
                }
            };

            let result_cstr = match CString::new(result_str) {
                Ok(s) => s,
                Err(e) => {
                    if !error_out.is_null() {
                        let msg = CString::new(format!("Failed to create result string: {}", e)).unwrap();
                        *error_out = msg.into_raw();
                    }
                    return SglErrorCode::MemoryError;
                }
            };

            *result_json_out = result_cstr.into_raw();
            if !error_out.is_null() {
                *error_out = ptr::null_mut();
            }
            SglErrorCode::Success
        }
        Err(e) => {
            if !error_out.is_null() {
                let msg = CString::new(format!("Parse error: {}", e)).unwrap();
                *error_out = msg.into_raw();
            }
            SglErrorCode::ParsingError
        }
    }
}

/// Parse tool calls incrementally from streaming chunks
///
/// # Arguments
/// * `handle` - Tool parser handle
/// * `chunk` - New text chunk from stream
/// * `tools_json` - JSON array of available tools (for validation, can be null/empty)
/// * `result_json_out` - Pointer to receive JSON result (must be freed with sgl_free_string)
/// * `error_out` - Optional pointer to receive error message
///
/// # Returns
/// * SglErrorCode::Success on success, error code on failure
///
/// # Result JSON Format
/// Same as parse_complete, but may have partial tool calls
#[no_mangle]
pub unsafe extern "C" fn sgl_tool_parser_parse_incremental(
    handle: *mut ToolParserHandle,
    chunk: *const c_char,
    tools_json: *const c_char,
    result_json_out: *mut *mut c_char,
    error_out: *mut *mut c_char,
) -> SglErrorCode {
    if handle.is_null() || chunk.is_null() || result_json_out.is_null() {
        if !error_out.is_null() {
            let msg = CString::new("Invalid arguments: null pointer").unwrap();
            *error_out = msg.into_raw();
        }
        return SglErrorCode::InvalidArgument;
    }

    let chunk_str = match CStr::from_ptr(chunk).to_str() {
        Ok(s) => s,
        Err(_) => {
            if !error_out.is_null() {
                let msg = CString::new("Invalid UTF-8 in chunk").unwrap();
                *error_out = msg.into_raw();
            }
            return SglErrorCode::InvalidArgument;
        }
    };

    // Parse tools JSON if provided
    let tools: Vec<Tool> = if !tools_json.is_null() {
        let tools_str = match CStr::from_ptr(tools_json).to_str() {
            Ok(s) => s,
            Err(_) => {
                if !error_out.is_null() {
                    let msg = CString::new("Invalid UTF-8 in tools_json").unwrap();
                    *error_out = msg.into_raw();
                }
                return SglErrorCode::InvalidArgument;
            }
        };
        match serde_json::from_str::<Vec<Tool>>(tools_str) {
            Ok(t) => t,
            Err(_) => vec![], // If parsing fails, use empty tools
        }
    } else {
        vec![]
    };

    let handle_ref = &*handle;
    let parser = Arc::clone(&handle_ref.parser);
    let model = handle_ref.model.clone();
    let history_count = handle_ref.history_tool_calls_count;

    // Use tokio runtime to run async code
    let result = RUNTIME.block_on(async {
        let mut parser_guard = parser.lock().await;
        parser_guard.parse_incremental(chunk_str, &tools).await
    });

    match result {
        Ok(streaming_result) => {
            // Convert StreamingParseResult to OpenAI format
            let handle_mut = &mut *handle;
            let openai_tool_calls: Vec<Value> = streaming_result
                .calls
                .into_iter()
                .map(|item| {
                    // For incremental parsing, we may not have complete tool calls yet
                    // Generate or reuse ID based on tool_index
                    let id = if let Some(ref name) = item.name {
                        // New tool call with name - generate ID and store it
                        let id = generate_tool_call_id(&model, name, item.tool_index, history_count);
                        handle_mut.tool_index_to_id.insert(item.tool_index, id.clone());
                        id
                    } else {
                        // Parameter update - reuse existing ID for this tool_index
                        handle_mut.tool_index_to_id
                            .get(&item.tool_index)
                            .cloned()
                            .unwrap_or_else(|| format!("call_{}", item.tool_index))
                    };

                    json!({
                        "id": id,
                        "type": "function",
                        "function": {
                            "name": item.name.unwrap_or_default(),
                            "arguments": item.parameters
                        }
                    })
                })
                .collect();

            // Build result JSON
            let result_json = json!({
                "normal_text": streaming_result.normal_text,
                "tool_calls": openai_tool_calls
            });

            let result_str = match serde_json::to_string(&result_json) {
                Ok(s) => s,
                Err(e) => {
                    if !error_out.is_null() {
                        let msg = CString::new(format!("Failed to serialize JSON: {}", e)).unwrap();
                        *error_out = msg.into_raw();
                    }
                    return SglErrorCode::ParsingError;
                }
            };

            let result_cstr = match CString::new(result_str) {
                Ok(s) => s,
                Err(e) => {
                    if !error_out.is_null() {
                        let msg = CString::new(format!("Failed to create result string: {}", e)).unwrap();
                        *error_out = msg.into_raw();
                    }
                    return SglErrorCode::MemoryError;
                }
            };

            *result_json_out = result_cstr.into_raw();
            if !error_out.is_null() {
                *error_out = ptr::null_mut();
            }
            SglErrorCode::Success
        }
        Err(e) => {
            if !error_out.is_null() {
                let msg = CString::new(format!("Parse incremental error: {}", e)).unwrap();
                *error_out = msg.into_raw();
            }
            SglErrorCode::ParsingError
        }
    }
}

/// Reset the parser state for reuse
#[no_mangle]
pub unsafe extern "C" fn sgl_tool_parser_reset(handle: *mut ToolParserHandle) {
    if handle.is_null() {
        return;
    }

    let handle_ref = &mut *handle;
    let parser = Arc::clone(&handle_ref.parser);

    // Reset parser state
    RUNTIME.block_on(async {
        let mut parser_guard = parser.lock().await;
        parser_guard.reset();
    });

    // Reset history count and tool index mapping
    handle_ref.history_tool_calls_count = 0;
    handle_ref.tool_index_to_id.clear();
}

/// Free a tool parser handle
#[no_mangle]
pub unsafe extern "C" fn sgl_tool_parser_free(handle: *mut ToolParserHandle) {
    if !handle.is_null() {
        let _ = Box::from_raw(handle);
    }
}

/// Helper function to generate tool call ID (matches router implementation)
fn generate_tool_call_id(
    model: &str,
    function_name: &str,
    index: usize,
    history_tool_calls_count: usize,
) -> String {
    use uuid::Uuid;
    
    if model.to_lowercase().contains("kimi") {
        // KimiK2 format: functions.{name}:{global_index}
        format!("functions.{}:{}", function_name, history_tool_calls_count + index)
    } else {
        // Standard OpenAI format: call_{24-char-uuid}
        format!("call_{}", &Uuid::new_v4().simple().to_string()[..24])
    }
}

// ============================================================================
// gRPC Response to OpenAI Format Conversion FFI
// ============================================================================

/// Handle for gRPC response converter (maintains state for streaming)
#[repr(C)]
pub struct GrpcResponseConverterHandle {
    tokenizer: Arc<dyn Tokenizer>,
    tool_parser: Option<Arc<tokio::sync::Mutex<Box<dyn ToolParser>>>>,
    stop_decoder: Option<Arc<tokio::sync::Mutex<StopSequenceDecoder>>>,
    model: String,
    request_id: String,
    created: u64,
    system_fingerprint: Option<String>,
    tools: Option<Vec<Tool>>,
    tool_choice: Option<ToolChoice>,
    history_tool_calls_count: usize,
    stream_buffers: HashMap<u32, String>, // Per-index text buffers
    has_tool_calls: HashMap<u32, bool>, // Track if tool calls were emitted
    is_first_chunk: HashMap<u32, bool>, // Track first chunk per index
}

/// Create a gRPC response converter handle
///
/// # Arguments
/// * `tokenizer_handle` - Tokenizer handle (must be valid)
/// * `model` - Model name
/// * `request_id` - Request ID
/// * `tools_json` - Optional JSON array of tools
/// * `tool_choice_json` - Optional JSON object for tool_choice
/// * `stop` - Optional stop sequences (JSON array)
/// * `stop_token_ids` - Optional stop token IDs (JSON array)
/// * `skip_special_tokens` - Whether to skip special tokens
/// * `error_out` - Optional pointer to receive error message
///
/// # Returns
/// * Pointer to GrpcResponseConverterHandle on success, null on failure
#[no_mangle]
pub unsafe extern "C" fn sgl_grpc_response_converter_create(
    tokenizer_handle: *mut TokenizerHandle,
    model: *const c_char,
    request_id: *const c_char,
    tools_json: *const c_char,
    tool_choice_json: *const c_char,
    stop: *const c_char,
    stop_token_ids: *const c_char,
    skip_special_tokens: c_int,
    error_out: *mut *mut c_char,
) -> *mut GrpcResponseConverterHandle {
    if tokenizer_handle.is_null() || model.is_null() || request_id.is_null() {
        if !error_out.is_null() {
            let msg = CString::new("Invalid arguments: null pointer").unwrap();
            *error_out = msg.into_raw();
        }
        return ptr::null_mut();
    }

    let model_str = match CStr::from_ptr(model).to_str() {
        Ok(s) => s,
        Err(_) => {
            if !error_out.is_null() {
                let msg = CString::new("Invalid UTF-8 in model").unwrap();
                *error_out = msg.into_raw();
            }
            return ptr::null_mut();
        }
    };

    let request_id_str = match CStr::from_ptr(request_id).to_str() {
        Ok(s) => s,
        Err(_) => {
            if !error_out.is_null() {
                let msg = CString::new("Invalid UTF-8 in request_id").unwrap();
                *error_out = msg.into_raw();
            }
            return ptr::null_mut();
        }
    };

    let handle_ref = &*tokenizer_handle;
    let tokenizer = Arc::clone(&handle_ref.tokenizer);

    // Parse tools if provided
    let tools: Option<Vec<Tool>> = if !tools_json.is_null() {
        match CStr::from_ptr(tools_json).to_str() {
            Ok(s) => serde_json::from_str::<Vec<Tool>>(s).ok(),
            Err(_) => None,
        }
    } else {
        None
    };

    // Parse tool_choice if provided
    let tool_choice: Option<ToolChoice> = if !tool_choice_json.is_null() {
        match CStr::from_ptr(tool_choice_json).to_str() {
            Ok(s) => serde_json::from_str::<ToolChoice>(s).ok(),
            Err(_) => None,
        }
    } else {
        None
    };

    // Parse stop sequences
    let stop: Option<StringOrArray> = if !stop.is_null() {
        let stop_str = match CStr::from_ptr(stop).to_str() {
            Ok(s) => s,
            Err(_) => return ptr::null_mut(),
        };
        serde_json::from_str::<StringOrArray>(stop_str).ok()
    } else {
        None
    };

    // Parse stop token IDs
    let stop_token_ids: Option<Vec<u32>> = if !stop_token_ids.is_null() {
        let ids_str = match CStr::from_ptr(stop_token_ids).to_str() {
            Ok(s) => s,
            Err(_) => return ptr::null_mut(),
        };
        serde_json::from_str::<Vec<u32>>(ids_str).ok()
    } else {
        None
    };

    // Create stop decoder if needed
    let stop_decoder = if stop.is_some() || stop_token_ids.is_some() {
        Some(Arc::new(tokio::sync::Mutex::new(
            crate::routers::grpc::utils::create_stop_decoder(
                &tokenizer,
                stop.as_ref(),
                stop_token_ids.as_ref(),
                skip_special_tokens != 0,
                false, // no_stop_trim
            ),
        )))
    } else {
        None
    };

    // Create tool parser if tools are provided
    let tool_parser = if tools.is_some() {
        PARSER_FACTORY.registry().create_for_model(model_str)
            .map(|p| Arc::new(tokio::sync::Mutex::new(p)))
    } else {
        None
    };

    // Get system fingerprint from model (simplified)
    let system_fingerprint = Some("fp_placeholder".to_string()); // TODO: Get actual fingerprint

    Box::into_raw(Box::new(GrpcResponseConverterHandle {
        tokenizer,
        tool_parser,
        stop_decoder,
        model: model_str.to_string(),
        request_id: request_id_str.to_string(),
        created: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        system_fingerprint,
        tools,
        tool_choice,
        history_tool_calls_count: 0,
        stream_buffers: HashMap::new(),
        has_tool_calls: HashMap::new(),
        is_first_chunk: HashMap::new(),
    }))
}

/// Convert a gRPC GenerateResponse chunk to OpenAI format
///
/// # Arguments
/// * `handle` - Converter handle
/// * `response_json` - JSON string of proto.GenerateResponse
/// * `result_json_out` - Pointer to receive OpenAI format JSON (must be freed with sgl_free_string)
/// * `error_out` - Optional pointer to receive error message
///
/// # Returns
/// * SglErrorCode::Success on success, error code on failure
#[no_mangle]
pub unsafe extern "C" fn sgl_grpc_response_converter_convert_chunk(
    handle: *mut GrpcResponseConverterHandle,
    response_json: *const c_char,
    result_json_out: *mut *mut c_char,
    error_out: *mut *mut c_char,
) -> SglErrorCode {
    if handle.is_null() || response_json.is_null() || result_json_out.is_null() {
        if !error_out.is_null() {
            let msg = CString::new("Invalid arguments: null pointer").unwrap();
            *error_out = msg.into_raw();
        }
        return SglErrorCode::InvalidArgument;
    }

    let response_str = match CStr::from_ptr(response_json).to_str() {
        Ok(s) => s,
        Err(_) => {
            if !error_out.is_null() {
                let msg = CString::new("Invalid UTF-8 in response_json").unwrap();
                *error_out = msg.into_raw();
            }
            return SglErrorCode::InvalidArgument;
        }
    };

    // Parse proto.GenerateResponse from JSON
    // Since prost-generated types don't support JSON deserialization directly,
    // we need to use a workaround. For now, we'll use a simplified approach:
    // The Go client should pass the response in a format that can be converted.
    // TODO: Consider using pbjson or implementing custom deserialization
    // For now, we'll create a minimal response structure from JSON
    let json_value: Value = match serde_json::from_str(response_str) {
        Ok(v) => v,
        Err(e) => {
            if !error_out.is_null() {
                let msg = CString::new(format!("Failed to parse response JSON: {}", e)).unwrap();
                *error_out = msg.into_raw();
            }
            return SglErrorCode::ParsingError;
        }
    };
    
    // Build proto::GenerateResponse from JSON value
    // This is a simplified conversion - in production, you might want a more robust solution
    let mut proto_response = proto::GenerateResponse {
        request_id: json_value.get("request_id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        response: None,
    };
    
    // Parse the response oneof field
    if let Some(chunk_json) = json_value.get("chunk") {
        // Parse GenerateStreamChunk
        let chunk = proto::GenerateStreamChunk {
            token_ids: chunk_json.get("token_ids")
                .and_then(|v| v.as_array())
                .map(|arr| arr.iter().filter_map(|v| v.as_u64().map(|n| n as u32)).collect())
                .unwrap_or_default(),
            prompt_tokens: chunk_json.get("prompt_tokens")
                .and_then(|v| v.as_i64())
                .map(|n| n as i32)
                .unwrap_or(0),
            completion_tokens: chunk_json.get("completion_tokens")
                .and_then(|v| v.as_i64())
                .map(|n| n as i32)
                .unwrap_or(0),
            cached_tokens: chunk_json.get("cached_tokens")
                .and_then(|v| v.as_i64())
                .map(|n| n as i32)
                .unwrap_or(0),
            output_logprobs: None,
            hidden_states: vec![],
            input_logprobs: None,
            index: 0,
        };
        proto_response.response = Some(proto::generate_response::Response::Chunk(chunk));
    } else if let Some(complete_json) = json_value.get("complete") {
        // Parse GenerateComplete
        let complete = proto::GenerateComplete {
            output_ids: complete_json.get("output_ids")
                .and_then(|v| v.as_array())
                .map(|arr| arr.iter().filter_map(|v| v.as_u64().map(|n| n as u32)).collect())
                .unwrap_or_default(),
            finish_reason: complete_json.get("finish_reason")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            prompt_tokens: complete_json.get("prompt_tokens")
                .and_then(|v| v.as_i64())
                .map(|n| n as i32)
                .unwrap_or(0),
            completion_tokens: complete_json.get("completion_tokens")
                .and_then(|v| v.as_i64())
                .map(|n| n as i32)
                .unwrap_or(0),
            cached_tokens: complete_json.get("cached_tokens")
                .and_then(|v| v.as_i64())
                .map(|n| n as i32)
                .unwrap_or(0),
            output_logprobs: None,
            all_hidden_states: vec![],
            input_logprobs: None,
            matched_stop: None,
            index: 0,
        };
        proto_response.response = Some(proto::generate_response::Response::Complete(complete));
    } else if let Some(error_json) = json_value.get("error") {
        // Parse GenerateError
        let error = proto::GenerateError {
            message: error_json.get("message")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            http_status_code: error_json.get("http_status_code")
                .and_then(|v| v.as_str())
                .unwrap_or("500")
                .to_string(),
            details: error_json.get("details")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
        };
        proto_response.response = Some(proto::generate_response::Response::Error(error));
    } else {
        if !error_out.is_null() {
            let msg = CString::new("Response JSON must contain 'chunk', 'complete', or 'error' field").unwrap();
            *error_out = msg.into_raw();
        }
        return SglErrorCode::ParsingError;
    }

    let handle_ref = &mut *handle;
    let tokenizer = Arc::clone(&handle_ref.tokenizer);
    let model = handle_ref.model.clone();
    let request_id = handle_ref.request_id.clone();
    let created = handle_ref.created;
    let system_fingerprint = handle_ref.system_fingerprint.clone();

    // Use tokio runtime to run async code
    let result = RUNTIME.block_on(async {
        convert_proto_chunk_to_openai(
            proto_response,
            handle_ref,
            &tokenizer,
            &model,
            &request_id,
            created,
            system_fingerprint.as_deref(),
        )
        .await
    });

    match result {
        Ok(Some(openai_response)) => {
            // Serialize to JSON
            let result_str = match serde_json::to_string(&openai_response) {
                Ok(s) => s,
                Err(e) => {
                    if !error_out.is_null() {
                        let msg = CString::new(format!("Failed to serialize response: {}", e)).unwrap();
                        *error_out = msg.into_raw();
                    }
                    return SglErrorCode::ParsingError;
                }
            };

            let result_cstr = match CString::new(result_str) {
                Ok(s) => s,
                Err(e) => {
                    if !error_out.is_null() {
                        let msg = CString::new(format!("Failed to create result string: {}", e)).unwrap();
                        *error_out = msg.into_raw();
                    }
                    return SglErrorCode::MemoryError;
                }
            };

            *result_json_out = result_cstr.into_raw();
            if !error_out.is_null() {
                *error_out = ptr::null_mut();
            }
            SglErrorCode::Success
        }
        Ok(None) => {
            // No response to send (e.g., empty chunk)
            let empty = CString::new("").unwrap();
            *result_json_out = empty.into_raw();
            if !error_out.is_null() {
                *error_out = ptr::null_mut();
            }
            SglErrorCode::Success
        }
        Err(e) => {
            if !error_out.is_null() {
                let msg = CString::new(format!("Conversion error: {}", e)).unwrap();
                *error_out = msg.into_raw();
            }
            SglErrorCode::ParsingError
        }
    }
}

/// Helper function to convert proto chunk to OpenAI format
async fn convert_proto_chunk_to_openai(
    proto_response: proto::GenerateResponse,
    handle: &mut GrpcResponseConverterHandle,
    tokenizer: &Arc<dyn Tokenizer>,
    model: &str,
    request_id: &str,
    created: u64,
    system_fingerprint: Option<&str>,
) -> Result<Option<crate::protocols::chat::ChatCompletionStreamResponse>, String> {
    use proto::generate_response::Response::*;
    use crate::protocols::chat::{ChatCompletionStreamResponse, ChatMessageDelta, ChatStreamChoice};

    match proto_response.response {
        Some(Chunk(chunk)) => {
            let index = chunk.index;

            // Mark as not first chunk if we've seen this index before
            let is_first = handle.is_first_chunk.entry(index).or_insert(true);
            let first_chunk = *is_first;
            *is_first = false;

            // Process tokens through stop decoder if available
            let chunk_text = if let Some(ref stop_decoder) = handle.stop_decoder {
                let mut decoder_guard = stop_decoder.lock().await;
                let mut text = String::new();
                for &token_id in &chunk.token_ids {
                    match decoder_guard.process_token(token_id).unwrap_or_else(|_| {
                        crate::tokenizer::stop::SequenceDecoderOutput::Held
                    }) {
                        crate::tokenizer::stop::SequenceDecoderOutput::Text(t) => {
                            text.push_str(&t);
                        }
                        crate::tokenizer::stop::SequenceDecoderOutput::StoppedWithText(t) => {
                            text.push_str(&t);
                            break;
                        }
                        crate::tokenizer::stop::SequenceDecoderOutput::Stopped => {
                            break;
                        }
                        crate::tokenizer::stop::SequenceDecoderOutput::Held => {}
                    }
                }
                text
            } else {
                // Decode tokens directly
                tokenizer.decode(&chunk.token_ids, true).unwrap_or_default()
            };

            if chunk_text.is_empty() {
                return Ok(None);
            }

            // Send first chunk with role
            if first_chunk {
                let first_response = ChatCompletionStreamResponse {
                    id: request_id.to_string(),
                    object: "chat.completion.chunk".to_string(),
                    created,
                    model: model.to_string(),
                    system_fingerprint: system_fingerprint.map(|s| s.to_string()),
                    choices: vec![ChatStreamChoice {
                        index,
                        delta: ChatMessageDelta {
                            role: Some("assistant".to_string()),
                            content: None,
                            tool_calls: None,
                            reasoning_content: None,
                        },
                        logprobs: None,
                        finish_reason: None,
                        matched_stop: None,
                    }],
                    usage: None,
                };
                return Ok(Some(first_response));
            }

            // Update stream buffer
            let stream_buffer = handle.stream_buffers.entry(index).or_default();
            stream_buffer.push_str(&chunk_text);

            // Handle tool calls if tools are provided
            if let (Some(ref tools), Some(ref tool_parser)) = (handle.tools.as_ref(), handle.tool_parser.as_ref()) {
                let tool_choice_enabled = !matches!(
                    handle.tool_choice,
                    Some(ToolChoice::Value(ToolChoiceValue::None))
                );

                if tool_choice_enabled {
                    let mut parser_guard = tool_parser.lock().await;
                    match parser_guard.parse_incremental(&chunk_text, tools).await {
                        Ok(streaming_result) => {
                            if !streaming_result.calls.is_empty() {
                                handle.has_tool_calls.insert(index, true);
                                // Convert tool call items to OpenAI format
                                let tool_call_deltas: Vec<_> = streaming_result
                                    .calls
                                    .into_iter()
                                    .map(|item| {
                                        let id = if let Some(ref name) = item.name {
                                            generate_tool_call_id(
                                                model,
                                                name,
                                                item.tool_index,
                                                handle.history_tool_calls_count,
                                            )
                                        } else {
                                            format!("call_{}", item.tool_index)
                                        };

                                        ToolCallDelta {
                                            index: item.tool_index as u32,
                                            id: Some(id),
                                            tool_type: if item.name.is_some() {
                                                Some("function".to_string())
                                            } else {
                                                None
                                            },
                                            function: Some(FunctionCallDelta {
                                                name: item.name,
                                                arguments: if !item.parameters.is_empty() {
                                                    Some(item.parameters)
                                                } else {
                                                    None
                                                },
                                            }),
                                        }
                                    })
                                    .collect();

                                let tool_response = ChatCompletionStreamResponse {
                                    id: request_id.to_string(),
                                    object: "chat.completion.chunk".to_string(),
                                    created,
                                    model: model.to_string(),
                                    system_fingerprint: system_fingerprint.map(|s| s.to_string()),
                                    choices: vec![ChatStreamChoice {
                                        index,
                                        delta: ChatMessageDelta {
                                            role: Some("assistant".to_string()),
                                            content: None,
                                            tool_calls: Some(tool_call_deltas),
                                            reasoning_content: None,
                                        },
                                        logprobs: None,
                                        finish_reason: None,
                                        matched_stop: None,
                                    }],
                                    usage: None,
                                };
                                return Ok(Some(tool_response));
                            }
                        }
                        Err(e) => {
                            // Log error but continue with regular content
                            tracing::warn!("Tool parser error: {}", e);
                        }
                    }
                }
            }

            // Regular content emission
            let content_response = ChatCompletionStreamResponse {
                id: request_id.to_string(),
                object: "chat.completion.chunk".to_string(),
                created,
                model: model.to_string(),
                system_fingerprint: system_fingerprint.map(|s| s.to_string()),
                choices: vec![ChatStreamChoice {
                    index,
                    delta: ChatMessageDelta {
                        role: Some("assistant".to_string()),
                        content: Some(chunk_text),
                        tool_calls: None,
                        reasoning_content: None,
                    },
                    logprobs: None,
                    finish_reason: None,
                    matched_stop: None,
                }],
                usage: None,
            };

            Ok(Some(content_response))
        }
        Some(Complete(complete)) => {
            let index = complete.index;

            // Flush any remaining text
            let final_text = handle.stream_buffers.remove(&index).unwrap_or_default();

            // Determine finish reason
            let finish_reason = if handle.has_tool_calls.get(&index).copied().unwrap_or(false)
                && complete.finish_reason == "stop"
            {
                "tool_calls".to_string()
            } else {
                complete.finish_reason.clone()
            };

            // Extract matched_stop
            let matched_stop = match &complete.matched_stop {
                Some(proto::generate_complete::MatchedStop::MatchedTokenId(token_id)) => {
                    Some(Value::Number(serde_json::Number::from(*token_id)))
                }
                Some(proto::generate_complete::MatchedStop::MatchedStopStr(stop_str)) => {
                    Some(Value::String(stop_str.clone()))
                }
                None => None,
            };

            // Build usage if available
            let usage = if complete.prompt_tokens > 0 || complete.completion_tokens > 0 {
                Some(Usage {
                    prompt_tokens: complete.prompt_tokens as u32,
                    completion_tokens: complete.completion_tokens as u32,
                    total_tokens: (complete.prompt_tokens + complete.completion_tokens) as u32,
                    completion_tokens_details: None,
                })
            } else {
                None
            };

            let finish_response = ChatCompletionStreamResponse {
                id: request_id.to_string(),
                object: "chat.completion.chunk".to_string(),
                created,
                model: model.to_string(),
                system_fingerprint: system_fingerprint.map(|s| s.to_string()),
                choices: vec![ChatStreamChoice {
                    index,
                    delta: ChatMessageDelta {
                        role: Some("assistant".to_string()),
                        content: if !final_text.is_empty() {
                            Some(final_text)
                        } else {
                            None
                        },
                        tool_calls: None,
                        reasoning_content: None,
                    },
                    logprobs: None,
                    finish_reason: Some(finish_reason),
                    matched_stop,
                }],
                usage,
            };

            Ok(Some(finish_response))
        }
        Some(Error(error)) => {
            Err(format!("Server error: {} (status: {})", error.message, error.http_status_code))
        }
        None => Ok(None),
    }
}

/// Free a gRPC response converter handle
#[no_mangle]
pub unsafe extern "C" fn sgl_grpc_response_converter_free(handle: *mut GrpcResponseConverterHandle) {
    if !handle.is_null() {
        let _ = Box::from_raw(handle);
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Convert proto::GenerateResponse to JSON Value (since prost types don't support serde)
fn proto_response_to_json(response: &proto::GenerateResponse) -> String {
    let mut json_obj = serde_json::Map::new();
    json_obj.insert("request_id".to_string(), Value::String(response.request_id.clone()));
    
    match &response.response {
        Some(proto::generate_response::Response::Chunk(chunk)) => {
            let mut chunk_obj = serde_json::Map::new();
            chunk_obj.insert("token_ids".to_string(), 
                Value::Array(chunk.token_ids.iter().map(|&id| Value::Number(id.into())).collect()));
            chunk_obj.insert("prompt_tokens".to_string(), Value::Number(chunk.prompt_tokens.into()));
            chunk_obj.insert("completion_tokens".to_string(), Value::Number(chunk.completion_tokens.into()));
            chunk_obj.insert("cached_tokens".to_string(), Value::Number(chunk.cached_tokens.into()));
            chunk_obj.insert("index".to_string(), Value::Number(chunk.index.into()));
            json_obj.insert("chunk".to_string(), Value::Object(chunk_obj));
        }
        Some(proto::generate_response::Response::Complete(complete)) => {
            let mut complete_obj = serde_json::Map::new();
            complete_obj.insert("output_ids".to_string(), 
                Value::Array(complete.output_ids.iter().map(|&id| Value::Number(id.into())).collect()));
            complete_obj.insert("finish_reason".to_string(), Value::String(complete.finish_reason.clone()));
            complete_obj.insert("prompt_tokens".to_string(), Value::Number(complete.prompt_tokens.into()));
            complete_obj.insert("completion_tokens".to_string(), Value::Number(complete.completion_tokens.into()));
            complete_obj.insert("cached_tokens".to_string(), Value::Number(complete.cached_tokens.into()));
            complete_obj.insert("index".to_string(), Value::Number(complete.index.into()));
            json_obj.insert("complete".to_string(), Value::Object(complete_obj));
        }
        Some(proto::generate_response::Response::Error(err)) => {
            let mut error_obj = serde_json::Map::new();
            error_obj.insert("message".to_string(), Value::String(err.message.clone()));
            error_obj.insert("http_status_code".to_string(), Value::String(err.http_status_code.clone()));
            error_obj.insert("details".to_string(), Value::String(err.details.clone()));
            json_obj.insert("error".to_string(), Value::Object(error_obj));
        }
        None => {}
    }
    
    serde_json::to_string(&Value::Object(json_obj)).unwrap_or_else(|_| "{}".to_string())
}

// ============================================================================
// Complete Request-Response Flow FFI (Client SDK)
// ============================================================================

/// Handle for complete client SDK (gRPC client + tokenizer)
/// This handle manages the connection to sglang and provides a complete SDK interface
pub struct SglangClientHandle {
    client: Arc<SglangSchedulerClient>,
    tokenizer: Arc<dyn Tokenizer>,
}

/// Handle for an active streaming request
/// This handle manages the stream and response converter
pub struct SglangStreamHandle {
    stream: Arc<tokio::sync::Mutex<AbortOnDropStream>>,
    converter: Arc<tokio::sync::Mutex<GrpcResponseConverterHandle>>,
    client: Arc<SglangSchedulerClient>,
}

/// Create a new SGLang client handle
///
/// # Arguments
/// * `endpoint` - gRPC endpoint (e.g., "grpc://localhost:20000")
/// * `tokenizer_path` - Path to tokenizer directory
/// * `error_out` - Optional pointer to receive error message
///
/// # Returns
/// * Pointer to SglangClientHandle on success, null on failure
#[no_mangle]
pub unsafe extern "C" fn sgl_client_create(
    endpoint: *const c_char,
    tokenizer_path: *const c_char,
    error_out: *mut *mut c_char,
) -> *mut SglangClientHandle {
    if endpoint.is_null() || tokenizer_path.is_null() {
        if !error_out.is_null() {
            let msg = CString::new("Invalid arguments: null pointer").unwrap();
            *error_out = msg.into_raw();
        }
        return ptr::null_mut();
    }

    let endpoint_str = match CStr::from_ptr(endpoint).to_str() {
        Ok(s) => s,
        Err(_) => {
            if !error_out.is_null() {
                let msg = CString::new("Invalid UTF-8 in endpoint").unwrap();
                *error_out = msg.into_raw();
            }
            return ptr::null_mut();
        }
    };

    let tokenizer_path_str = match CStr::from_ptr(tokenizer_path).to_str() {
        Ok(s) => s,
        Err(_) => {
            if !error_out.is_null() {
                let msg = CString::new("Invalid UTF-8 in tokenizer_path").unwrap();
                *error_out = msg.into_raw();
            }
            return ptr::null_mut();
        }
    };

    // Create tokenizer
    let tokenizer = match create_tokenizer_from_file(tokenizer_path_str) {
        Ok(t) => t,
        Err(e) => {
            if !error_out.is_null() {
                let msg = CString::new(format!("Failed to create tokenizer: {}", e)).unwrap();
                *error_out = msg.into_raw();
            }
            return ptr::null_mut();
        }
    };

    // Create gRPC client
    let client = match RUNTIME.block_on(async {
        SglangSchedulerClient::connect(endpoint_str).await
    }) {
        Ok(c) => Arc::new(c),
        Err(e) => {
            if !error_out.is_null() {
                let msg = CString::new(format!("Failed to connect to endpoint: {}", e)).unwrap();
                *error_out = msg.into_raw();
            }
            return ptr::null_mut();
        }
    };

    Box::into_raw(Box::new(SglangClientHandle {
        client,
        tokenizer,
    }))
}

/// Free a client handle
#[no_mangle]
pub unsafe extern "C" fn sgl_client_free(handle: *mut SglangClientHandle) {
    if !handle.is_null() {
        let _ = Box::from_raw(handle);
    }
}

/// Send a chat completion request and start streaming
///
/// # Arguments
/// * `client_handle` - Client handle
/// * `request_json` - OpenAI ChatCompletionRequest as JSON string
/// * `stream_handle_out` - Pointer to receive stream handle
/// * `error_out` - Optional pointer to receive error message
///
/// # Returns
/// * SglErrorCode::Success on success, error code on failure
#[no_mangle]
pub unsafe extern "C" fn sgl_client_chat_completion_stream(
    client_handle: *mut SglangClientHandle,
    request_json: *const c_char,
    stream_handle_out: *mut *mut SglangStreamHandle,
    error_out: *mut *mut c_char,
) -> SglErrorCode {
    if client_handle.is_null() || request_json.is_null() || stream_handle_out.is_null() {
        if !error_out.is_null() {
            let msg = CString::new("Invalid arguments: null pointer").unwrap();
            *error_out = msg.into_raw();
        }
        return SglErrorCode::InvalidArgument;
    }

    let request_str = match CStr::from_ptr(request_json).to_str() {
        Ok(s) => s,
        Err(_) => {
            if !error_out.is_null() {
                let msg = CString::new("Invalid UTF-8 in request_json").unwrap();
                *error_out = msg.into_raw();
            }
            return SglErrorCode::InvalidArgument;
        }
    };

    let client_ref = &*client_handle;
    let client = Arc::clone(&client_ref.client);
    let tokenizer = Arc::clone(&client_ref.tokenizer);

    // Parse OpenAI ChatCompletionRequest
    let chat_request: ChatCompletionRequest = match serde_json::from_str(request_str) {
        Ok(req) => req,
        Err(e) => {
            if !error_out.is_null() {
                let msg = CString::new(format!("Failed to parse request JSON: {}", e)).unwrap();
                *error_out = msg.into_raw();
            }
            return SglErrorCode::ParsingError;
        }
    };

    // Process messages and apply chat template
    let processed_messages = match process_chat_messages(&chat_request, tokenizer.as_ref()) {
        Ok(msgs) => msgs,
        Err(e) => {
            if !error_out.is_null() {
                let msg = CString::new(format!("Failed to process messages: {}", e)).unwrap();
                *error_out = msg.into_raw();
            }
            return SglErrorCode::TokenizationError;
        }
    };

    // Tokenize
    let token_ids = match tokenizer.encode(&processed_messages.text) {
        Ok(encoding) => encoding.token_ids().to_vec(),
        Err(e) => {
            if !error_out.is_null() {
                let msg = CString::new(format!("Failed to tokenize: {}", e)).unwrap();
                *error_out = msg.into_raw();
            }
            return SglErrorCode::TokenizationError;
        }
    };

    // Generate tool constraints if needed
    let tool_constraint = if let Some(tools) = chat_request.tools.as_ref() {
        match generate_tool_constraints(tools, &chat_request.tool_choice, &chat_request.model) {
            Ok(Some((constraint_type, constraint_value))) => Some((constraint_type, constraint_value)),
            Ok(None) => None,
            Err(e) => {
                if !error_out.is_null() {
                    let msg = CString::new(format!("Failed to generate tool constraints: {}", e)).unwrap();
                    *error_out = msg.into_raw();
                }
                return SglErrorCode::ParsingError;
            }
        }
    } else {
        None
    };

    // Build GenerateRequest
    let request_id = format!("chatcmpl-{}", Uuid::new_v4());
    let proto_request = match client.build_generate_request_from_chat(
        request_id.clone(),
        &chat_request,
        processed_messages.text,
        token_ids,
        processed_messages.multimodal_inputs,
        tool_constraint,
    ) {
        Ok(req) => req,
        Err(e) => {
            if !error_out.is_null() {
                let msg = CString::new(format!("Failed to build generate request: {}", e)).unwrap();
                *error_out = msg.into_raw();
            }
            return SglErrorCode::ParsingError;
        }
    };

    // Send request and get stream
    let stream = match RUNTIME.block_on(async {
        client.generate(proto_request).await
    }) {
        Ok(s) => s,
        Err(e) => {
            if !error_out.is_null() {
                let msg = CString::new(format!("Failed to send request: {}", e)).unwrap();
                *error_out = msg.into_raw();
            }
            return SglErrorCode::UnknownError;
        }
    };

    // Create response converter
    let tools_json = chat_request.tools.as_ref()
        .and_then(|t| serde_json::to_string(t).ok())
        .map(|s| CString::new(s).unwrap().into_raw());
    let tool_choice_json = chat_request.tool_choice.as_ref()
        .and_then(|tc| serde_json::to_string(tc).ok())
        .map(|s| CString::new(s).unwrap().into_raw());
    let stop_json = chat_request.stop.as_ref()
        .and_then(|s| serde_json::to_string(s).ok())
        .map(|s| CString::new(s).unwrap().into_raw());
    let stop_token_ids_json = chat_request.stop_token_ids.as_ref()
        .and_then(|ids| serde_json::to_string(ids).ok())
        .map(|s| CString::new(s).unwrap().into_raw());

    // Create tokenizer handle for converter (we'll create a temporary one)
    let tokenizer_handle = Box::into_raw(Box::new(TokenizerHandle {
        tokenizer: Arc::clone(&tokenizer),
    }));

    let converter = sgl_grpc_response_converter_create(
        tokenizer_handle,
        CString::new(chat_request.model.clone()).unwrap().as_ptr(),
        CString::new(request_id.clone()).unwrap().as_ptr(),
        tools_json.unwrap_or(ptr::null_mut()),
        tool_choice_json.unwrap_or(ptr::null_mut()),
        stop_json.unwrap_or(ptr::null_mut()),
        stop_token_ids_json.unwrap_or(ptr::null_mut()),
        if chat_request.skip_special_tokens { 1 } else { 0 },
        error_out,
    );

    // Free temporary tokenizer handle (converter now owns the tokenizer)
    let _ = Box::from_raw(tokenizer_handle);

    if converter.is_null() {
        return SglErrorCode::MemoryError;
    }

    // Clean up temporary CStrings
    if let Some(ptr) = tools_json {
        let _ = CString::from_raw(ptr);
    }
    if let Some(ptr) = tool_choice_json {
        let _ = CString::from_raw(ptr);
    }
    if let Some(ptr) = stop_json {
        let _ = CString::from_raw(ptr);
    }
    if let Some(ptr) = stop_token_ids_json {
        let _ = CString::from_raw(ptr);
    }

    // Create stream handle
    *stream_handle_out = Box::into_raw(Box::new(SglangStreamHandle {
        stream: Arc::new(tokio::sync::Mutex::new(stream)),
        converter: Arc::new(tokio::sync::Mutex::new(*Box::from_raw(converter))),
        client: Arc::clone(&client),
    }));

    SglErrorCode::Success
}

/// Read next chunk from stream and convert to OpenAI format
///
/// # Arguments
/// * `stream_handle` - Stream handle
/// * `response_json_out` - Pointer to receive OpenAI format JSON (must be freed with sgl_free_string)
/// * `is_done_out` - Pointer to receive 1 if stream is done, 0 otherwise
/// * `error_out` - Optional pointer to receive error message
///
/// # Returns
/// * SglErrorCode::Success on success, error code on failure
#[no_mangle]
pub unsafe extern "C" fn sgl_stream_read_next(
    stream_handle: *mut SglangStreamHandle,
    response_json_out: *mut *mut c_char,
    is_done_out: *mut c_int,
    error_out: *mut *mut c_char,
) -> SglErrorCode {
    if stream_handle.is_null() || response_json_out.is_null() || is_done_out.is_null() {
        if !error_out.is_null() {
            let msg = CString::new("Invalid arguments: null pointer").unwrap();
            *error_out = msg.into_raw();
        }
        return SglErrorCode::InvalidArgument;
    }

    let handle_ref = &*stream_handle;
    let stream = Arc::clone(&handle_ref.stream);
    let converter = Arc::clone(&handle_ref.converter);

    // Read next chunk from stream
    let chunk_result = RUNTIME.block_on(async {
        let mut stream_guard = stream.lock().await;
        stream_guard.next().await
    });

    match chunk_result {
        Some(Ok(proto_response)) => {
            // Convert proto response to OpenAI format
            // We need to get the converter lock first
            let conversion_result = RUNTIME.block_on(async {
                let mut converter_guard = converter.lock().await;
                
                // Clone necessary fields for conversion
                let tokenizer = Arc::clone(&converter_guard.tokenizer);
                let model = converter_guard.model.clone();
                let request_id = converter_guard.request_id.clone();
                let created = converter_guard.created;
                let system_fingerprint = converter_guard.system_fingerprint.clone();
                
                // Call the conversion function
                convert_proto_chunk_to_openai(
                    proto_response.clone(),
                    &mut *converter_guard,
                    &tokenizer,
                    &model,
                    &request_id,
                    created,
                    system_fingerprint.as_deref(),
                )
                .await
            });

            match conversion_result {
                Ok(Some(openai_response)) => {
                    // Serialize to JSON
                    let result_str = match serde_json::to_string(&openai_response) {
                        Ok(s) => s,
                        Err(e) => {
                            if !error_out.is_null() {
                                let msg = CString::new(format!("Failed to serialize response: {}", e)).unwrap();
                                *error_out = msg.into_raw();
                            }
                            return SglErrorCode::ParsingError;
                        }
                    };

                    let result_cstr = match CString::new(result_str) {
                        Ok(s) => s,
                        Err(e) => {
                            if !error_out.is_null() {
                                let msg = CString::new(format!("Failed to create result string: {}", e)).unwrap();
                                *error_out = msg.into_raw();
                            }
                            return SglErrorCode::MemoryError;
                        }
                    };

                    // Check if this is a complete response (stream done)
                    let is_complete = matches!(proto_response.response, Some(proto::generate_response::Response::Complete(_)) | Some(proto::generate_response::Response::Error(_)));

                    *response_json_out = result_cstr.into_raw();
                    *is_done_out = if is_complete { 1 } else { 0 };

                    if is_complete {
                        // Mark stream as completed
                        RUNTIME.block_on(async {
                            let stream_guard = stream.lock().await;
                            stream_guard.mark_completed();
                        });
                    }

                    SglErrorCode::Success
                }
                Ok(None) => {
                    // No response to send (e.g., empty chunk)
                    *response_json_out = ptr::null_mut();
                    *is_done_out = 0;
                    SglErrorCode::Success
                }
                Err(e) => {
                    if !error_out.is_null() {
                        let msg = CString::new(format!("Conversion error: {}", e)).unwrap();
                        *error_out = msg.into_raw();
                    }
                    SglErrorCode::ParsingError
                }
            }
        }
        Some(Err(e)) => {
            if !error_out.is_null() {
                let msg = CString::new(format!("Stream error: {}", e)).unwrap();
                *error_out = msg.into_raw();
            }
            *is_done_out = 1;
            SglErrorCode::UnknownError
        }
        None => {
            // Stream ended
            *response_json_out = ptr::null_mut();
            *is_done_out = 1;
            SglErrorCode::Success
        }
    }
}

/// Free a stream handle
#[no_mangle]
pub unsafe extern "C" fn sgl_stream_free(handle: *mut SglangStreamHandle) {
    if !handle.is_null() {
        let handle_ref = Box::from_raw(handle);
        // Free converter
        let converter = Arc::try_unwrap(handle_ref.converter)
            .ok()
            .map(|m| m.into_inner());
        if let Some(conv) = converter {
            sgl_grpc_response_converter_free(Box::into_raw(Box::new(conv)));
        }
    }
}

// ============================================================================
// Tool Constraint Generation FFI
// ============================================================================

/// Generate tool constraints JSON Schema
///
/// # Arguments
/// * `tools_json` - JSON array of tools
/// * `tool_choice_json` - JSON object representing tool_choice
/// * `constraint_type_out` - Pointer to receive constraint type (e.g., "json_schema")
/// * `constraint_schema_out` - Pointer to receive constraint schema JSON
/// * `error_out` - Optional pointer to receive error message
///
/// # Returns
/// * SglErrorCode::Success on success, error code on failure
///
/// # Note
/// Both constraint_type_out and constraint_schema_out must be freed with sgl_free_string.
#[no_mangle]
pub unsafe extern "C" fn sgl_generate_tool_constraints(
    _tools_json: *const c_char,
    _tool_choice_json: *const c_char,
    _constraint_type_out: *mut *mut c_char,
    _constraint_schema_out: *mut *mut c_char,
    error_out: *mut *mut c_char,
) -> SglErrorCode {
    // Implementation would parse JSON and call generate_tool_constraints
    // This is a placeholder
    if !error_out.is_null() {
        let msg = CString::new("Tool constraint generation not yet implemented in FFI").unwrap();
        *error_out = msg.into_raw();
    }
    SglErrorCode::UnknownError
}

// ============================================================================
// C Header Generation
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_codes() {
        assert_eq!(SglErrorCode::Success as i32, 0);
        assert_eq!(SglErrorCode::InvalidArgument as i32, 1);
    }
}

