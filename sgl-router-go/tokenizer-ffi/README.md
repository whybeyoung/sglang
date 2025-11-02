# Rust Tokenizer FFI

This directory contains a Rust FFI wrapper for the HuggingFace `tokenizers` crate,
allowing Go code to use Rust tokenizers via CGO.

## Building

```bash
# Install cbindgen for header generation (optional)
cargo install cbindgen

# Build the library
./build.sh

# Or manually:
cargo build --release
```

## Usage

The library provides C-compatible functions that can be called from Go via CGO.

### Functions

- `tokenizer_from_file(path)`: Initialize tokenizer from file
- `tokenizer_encode(text, output, capacity, output_len)`: Encode text to token IDs
- `tokenizer_decode(token_ids, len, skip_special, output, capacity, output_len)`: Decode token IDs to text
- `tokenizer_vocab_size()`: Get vocabulary size
- `tokenizer_token_to_id(token)`: Get token ID for a token string

## Integration with Go

See `internal/tokenizer/rust_ffi.go` for the Go wrapper implementation.

## Notes

- The tokenizer uses a global mutex for thread safety
- Memory management: C strings are allocated/deallocated properly
- Error handling via `TokenizerError` struct

## Limitations

- Decode functionality requires proper encoding objects (work in progress)
- Some tokenizer features may not be fully exposed via FFI
- Performance overhead from CGO boundary crossing

