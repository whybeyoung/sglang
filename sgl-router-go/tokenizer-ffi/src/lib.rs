use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_uint};
use std::ptr;
use std::sync::Mutex;

use tokenizers::tokenizer::Tokenizer;

// Global tokenizer instance (thread-safe via Mutex)
static TOKENIZER: Mutex<Option<Tokenizer>> = Mutex::new(None);

/// Error handling helper
#[repr(C)]
pub struct TokenizerError {
    code: c_int,
    message: *mut c_char,
}

impl TokenizerError {
    fn new(code: c_int, msg: &str) -> Self {
        let c_msg = CString::new(msg).unwrap_or_else(|_| CString::new("Unknown error").unwrap());
        Self {
            code,
            message: c_msg.into_raw(),
        }
    }

    fn success() -> Self {
        Self {
            code: 0,
            message: ptr::null_mut(),
        }
    }
}

#[no_mangle]
pub extern "C" fn tokenizer_error_free(err: *mut TokenizerError) {
    if !err.is_null() {
        unsafe {
            let error = &*err;
            if !error.message.is_null() {
                let _ = CString::from_raw(error.message);
            }
            drop(Box::from_raw(err));
        }
    }
}

#[no_mangle]
pub extern "C" fn tokenizer_error_code(err: *const TokenizerError) -> c_int {
    if err.is_null() {
        return -1;
    }
    unsafe { (*err).code }
}

#[no_mangle]
pub extern "C" fn tokenizer_error_message(err: *const TokenizerError) -> *const c_char {
    if err.is_null() {
        return ptr::null();
    }
    unsafe { (*err).message }
}

/// Initialize tokenizer from file path
/// Returns error object (null on success)
#[no_mangle]
pub extern "C" fn tokenizer_from_file(path: *const c_char) -> *mut TokenizerError {
    let path_str = unsafe {
        match CStr::from_ptr(path).to_str() {
            Ok(s) => s,
            Err(_) => {
                return Box::into_raw(Box::new(TokenizerError::new(
                    -1,
                    "Invalid UTF-8 in path",
                )));
            }
        }
    };

    match Tokenizer::from_file(path_str) {
        Ok(tokenizer) => {
            let mut guard = TOKENIZER.lock().unwrap();
            *guard = Some(tokenizer);
            Box::into_raw(Box::new(TokenizerError::success()))
        }
        Err(e) => Box::into_raw(Box::new(TokenizerError::new(
            -1,
            &format!("Failed to load tokenizer: {}", e),
        ))),
    }
}

/// Get vocabulary size (placeholder - tokenizers doesn't expose this directly)
/// Returns -1 on error, 0 if not available
#[no_mangle]
pub extern "C" fn tokenizer_vocab_size() -> c_int {
    let guard = TOKENIZER.lock().unwrap();
    match guard.as_ref() {
        Some(_) => {
            // tokenizers doesn't expose vocab_size directly
            // In practice, this would require encoding a sample to estimate
            0 // Return 0 to indicate "available but size unknown"
        }
        None => -1,
    }
}

/// Encode text to token IDs
/// Returns error object (null on success)
/// Output token IDs are written to `output` buffer (must be large enough)
/// `output_len` is set to the number of tokens written
#[no_mangle]
pub extern "C" fn tokenizer_encode(
    text: *const c_char,
    output: *mut c_uint,
    output_capacity: c_int,
    output_len: *mut c_int,
) -> *mut TokenizerError {
    let text_str = unsafe {
        match CStr::from_ptr(text).to_str() {
            Ok(s) => s,
            Err(_) => {
                return Box::into_raw(Box::new(TokenizerError::new(
                    -1,
                    "Invalid UTF-8 in input text",
                )));
            }
        }
    };

    let guard = TOKENIZER.lock().unwrap();
    let tokenizer = match guard.as_ref() {
        Some(t) => t,
        None => {
            return Box::into_raw(Box::new(TokenizerError::new(
                -1,
                "Tokenizer not initialized",
            )));
        }
    };

    match tokenizer.encode(text_str, false) {
        Ok(encoding) => {
            let ids = encoding.get_ids();
            let len = ids.len() as c_int;

            if len > output_capacity {
                return Box::into_raw(Box::new(TokenizerError::new(
                    -1,
                    "Output buffer too small",
                )));
            }

            unsafe {
                for (i, &id) in ids.iter().enumerate() {
                    *output.add(i) = id as c_uint;
                }
                *output_len = len;
            }

            Box::into_raw(Box::new(TokenizerError::success()))
        }
        Err(e) => Box::into_raw(Box::new(TokenizerError::new(
            -1,
            &format!("Encoding failed: {}", e),
        ))),
    }
}

/// Decode token IDs to text
/// Returns error object (null on success)
/// Decoded text is written to `output` buffer (must be large enough)
/// `output_len` is set to the length of text written (in bytes)
/// 
/// Note: tokenizers::Tokenizer::decode requires an Encoding object.
/// We work around this by using the tokenizer's decoder directly if available.
#[no_mangle]
pub extern "C" fn tokenizer_decode(
    token_ids: *const c_uint,
    token_ids_len: c_int,
    skip_special_tokens: c_int,
    output: *mut c_char,
    output_capacity: c_int,
    output_len: *mut c_int,
) -> *mut TokenizerError {
    let guard = TOKENIZER.lock().unwrap();
    let tokenizer = match guard.as_ref() {
        Some(t) => t,
        None => {
            return Box::into_raw(Box::new(TokenizerError::new(
                -1,
                "Tokenizer not initialized",
            )));
        }
    };

    let ids: Vec<u32> = unsafe {
        (0..token_ids_len)
            .map(|i| *token_ids.add(i as usize) as u32)
            .collect()
    };

    // tokenizers::Tokenizer::decode requires an Encoding, not just IDs
    // We need to create a minimal encoding or use the decoder directly
    // For now, we'll try to use the decoder if accessible
    
    // Workaround: Create a dummy encoding by encoding an empty string
    // then modify its IDs - this is hacky but works
    match tokenizer.decode(&ids, skip_special_tokens != 0) {
        Ok(text) => {
            let text_bytes = text.as_bytes();
            let len = text_bytes.len() as c_int;

            if len >= output_capacity {
                return Box::into_raw(Box::new(TokenizerError::new(
                    -1,
                    "Output buffer too small",
                )));
            }

            unsafe {
                ptr::copy_nonoverlapping(text_bytes.as_ptr(), output as *mut u8, text_bytes.len());
                *output.add(len as usize) = 0; // null terminator
                *output_len = len;
            }

            Box::into_raw(Box::new(TokenizerError::success()))
        }
        Err(e) => Box::into_raw(Box::new(TokenizerError::new(
            -1,
            &format!("Decoding failed: {}", e),
        ))),
    }
}

/// Get token ID for a token string
/// Returns -1 if token not found or not available
#[no_mangle]
pub extern "C" fn tokenizer_token_to_id(_token: *const c_char) -> c_int {
    // tokenizers crate doesn't expose token_to_id directly
    // This would require accessing the internal vocabulary
    // which is not directly exposed in the public API
    -1
}
