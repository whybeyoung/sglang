//! Stream handling FFI functions

use std::ffi::CString;
use std::os::raw::{c_char, c_int};
use std::ptr;
use std::sync::Arc;
use tokio::runtime::Runtime;
use once_cell::sync::Lazy;
use futures_util::StreamExt;

use crate::grpc_client::{proto, sglang_scheduler::{SglangSchedulerClient, AbortOnDropStream}};

use super::error::{SglErrorCode, set_error_message};
use super::grpc_converter::{GrpcResponseConverterHandle, convert_proto_chunk_to_openai};

/// Global tokio runtime for async operations
static RUNTIME: Lazy<Runtime> = Lazy::new(|| {
    Runtime::new().expect("Failed to create tokio runtime for stream FFI")
});

/// Handle for an active streaming request
/// This handle manages the stream and response converter
pub struct SglangStreamHandle {
    pub(crate) stream: Arc<tokio::sync::Mutex<AbortOnDropStream>>,
    pub(crate) converter: Arc<tokio::sync::Mutex<GrpcResponseConverterHandle>>,
    #[allow(dead_code)]
    pub(crate) client: Arc<SglangSchedulerClient>,
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
        set_error_message(error_out, "Invalid arguments: null pointer");
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
                            set_error_message(error_out, &format!("Failed to serialize response: {}", e));
                            return SglErrorCode::ParsingError;
                        }
                    };

                    let result_cstr = match CString::new(result_str) {
                        Ok(s) => s,
                        Err(e) => {
                            set_error_message(error_out, &format!("Failed to create result string: {}", e));
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
                    set_error_message(error_out, &format!("Conversion error: {}", e));
                    SglErrorCode::ParsingError
                }
            }
        }
        Some(Err(e)) => {
            set_error_message(error_out, &format!("Stream error: {}", e));
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
            super::grpc_converter::sgl_grpc_response_converter_free(Box::into_raw(Box::new(conv)));
        }
    }
}

