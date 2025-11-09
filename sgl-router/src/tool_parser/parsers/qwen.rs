use async_trait::async_trait;
use regex::Regex;
use serde_json::Value;

use crate::{
    protocols::common::Tool,
    tool_parser::{
        errors::{ParserError, ParserResult},
        parsers::helpers,
        partial_json::PartialJson,
        traits::ToolParser,
        types::{FunctionCall, StreamingParseResult, ToolCall},
    },
};

/// Qwen format parser for tool calls
///
/// Handles the Qwen 2.5/3 specific format:
/// `<tool_call>\n{"name": "func", "arguments": {...}}\n</tool_call>`
///
/// Features:
/// - Tool Call Tags: `<tool_call>` and `</tool_call>` wrap each individual call
/// - Each individual call is separated by `\n`
/// - Function Call Object: JSON object with "name" and "arguments" fields
///
/// Reference: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct?chat_template=default
pub struct QwenParser {
    /// Parser for handling incomplete JSON during streaming
    partial_json: PartialJson,

    /// Regex for extracting tool calls in parse_complete
    extractor: Regex,

    /// Buffer for accumulating incomplete patterns across chunks
    buffer: String,

    /// Stores complete tool call info (name and arguments) for each tool being parsed
    prev_tool_call_arr: Vec<Value>,

    /// Index of currently streaming tool call (-1 means no active tool)
    current_tool_id: i32,

    /// Flag for whether current tool's name has been sent to client
    current_tool_name_sent: bool,

    /// Tracks raw JSON string content streamed to client for each tool's arguments
    streamed_args_for_tool: Vec<String>,

    /// Buffer for normal text that might precede partial end tokens
    normal_text_buffer: String,

    /// Token configuration
    /// Start/end tokens for each individual tool call (not the entire sequence)
    individual_tool_start_token: &'static str,
    individual_tool_end_token: &'static str,
    tool_call_separator: &'static str,

    /// XML format streaming state
    /// Whether we're currently parsing an XML format tool call
    in_xml_tool_call: bool,
    /// Format detection: None = not detected yet, Some(true) = XML, Some(false) = JSON
    /// This avoids repeated format detection on every chunk
    format_detected: Option<bool>,
    /// Current function name for XML format
    xml_current_function_name: String,
    /// Current parameters for XML format (as JSON map)
    xml_current_parameters: serde_json::Map<String, Value>,
    /// Streamed parameters for XML format (for diff calculation)
    xml_streamed_parameters: serde_json::Map<String, Value>,
    /// Whether we're currently inside a parameter tag
    in_parameter: bool,
    /// Current parameter key being parsed
    current_parameter_key: String,
    /// Buffer for current parameter value (accumulated across chunks)
    current_parameter_value: String,
}

impl QwenParser {
    /// Create a new Qwen parser
    pub fn new() -> Self {
        // Use (?s) flag for DOTALL mode to handle newlines
        // Support both JSON format: <tool_call>\n{"name": "...", "arguments": {...}}\n</tool_call>
        // and XML format: <tool_call>\n<function=name>\n<parameter=key>value</parameter>\n</function>\n</tool_call>
        let pattern = r"(?s)<tool_call>\s*(.*?)\s*</tool_call>";
        let extractor = Regex::new(pattern).expect("Valid regex pattern");

        Self {
            partial_json: PartialJson::default(),
            extractor,
            buffer: String::new(),
            prev_tool_call_arr: Vec::new(),
            current_tool_id: -1,
            current_tool_name_sent: false,
            streamed_args_for_tool: Vec::new(),
            normal_text_buffer: String::new(),
            individual_tool_start_token: "<tool_call>\n",
            individual_tool_end_token: "\n</tool_call>",
            tool_call_separator: "\n",
            in_xml_tool_call: false,
            format_detected: None,
            xml_current_function_name: String::new(),
            xml_current_parameters: serde_json::Map::new(),
            xml_streamed_parameters: serde_json::Map::new(),
            in_parameter: false,
            current_parameter_key: String::new(),
            current_parameter_value: String::new(),
        }
    }

    /// Parse a single JSON object into a ToolCall
    fn parse_single_object(&self, obj: &Value) -> ParserResult<Option<ToolCall>> {
        let name = obj.get("name").and_then(|v| v.as_str());

        if let Some(name) = name {
            // Get arguments - Qwen uses "arguments" key
            let empty_obj = Value::Object(serde_json::Map::new());
            let args = obj.get("arguments").unwrap_or(&empty_obj);

            // Convert arguments to JSON string
            let arguments = serde_json::to_string(args)
                .map_err(|e| ParserError::ParsingFailed(e.to_string()))?;

            Ok(Some(ToolCall {
                function: FunctionCall {
                    name: name.to_string(),
                    arguments,
                },
            }))
        } else {
            Ok(None)
        }
    }

    /// Parse XML format tool call: <function=name><parameter=key>value</parameter></function>
    fn parse_xml_format(&self, content: &str) -> ParserResult<Option<ToolCall>> {
        use regex::Regex;
        
        // Pattern for function name: <function=name>
        let function_pattern = Regex::new(r"<function=([^>]+)>")
            .map_err(|e| ParserError::ParsingFailed(format!("Invalid regex: {}", e)))?;
        
        let function_captures = function_pattern.captures(content)
            .ok_or_else(|| ParserError::ParsingFailed("No function name found".to_string()))?;
        
        let function_name = function_captures.get(1)
            .ok_or_else(|| ParserError::ParsingFailed("Function name capture failed".to_string()))?
            .as_str()
            .trim()
            .to_string();

        if function_name.is_empty() {
            return Ok(None);
        }

        // Pattern for parameters: <parameter=key>value</parameter>
        let param_pattern = Regex::new(r"<parameter=([^>]+)>(.*?)</parameter>")
            .map_err(|e| ParserError::ParsingFailed(format!("Invalid regex: {}", e)))?;
        
        let mut parameters = serde_json::Map::new();

        for cap in param_pattern.captures_iter(content) {
            if let (Some(key_match), Some(value_match)) = (cap.get(1), cap.get(2)) {
                let key = key_match.as_str().trim().to_string();
                let value = value_match.as_str().trim();
                
                // Try to parse value as JSON, otherwise use as string
                match serde_json::from_str::<Value>(value) {
                    Ok(json_value) => {
                        parameters.insert(key, json_value);
                    }
                    Err(_) => {
                        // If not valid JSON, treat as string
                        parameters.insert(key, Value::String(value.to_string()));
                    }
                }
            }
        }

        let arguments = serde_json::to_string(&parameters)
            .map_err(|e| ParserError::ParsingFailed(e.to_string()))?;

        Ok(Some(ToolCall {
            function: FunctionCall {
                name: function_name,
                arguments,
            },
        }))
    }

    /// Detect if content is JSON or XML format
    fn detect_format(&self, content: &str) -> ToolCallFormat {
        // Check for XML format markers
        if content.contains("<function=") && content.contains("<parameter=") {
            ToolCallFormat::Xml
        } else if content.trim_start().starts_with('{') {
            ToolCallFormat::Json
        } else {
            ToolCallFormat::Unknown
        }
    }
}

/// Format of tool call content inside <tool_call> tags
enum ToolCallFormat {
    Json,
    Xml,
    Unknown,
}

impl Default for QwenParser {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ToolParser for QwenParser {
    async fn parse_complete(&self, text: &str) -> ParserResult<(String, Vec<ToolCall>)> {
        // Check if text contains Qwen format
        if !self.has_tool_markers(text) {
            return Ok((text.to_string(), vec![]));
        }

        // Find where the first tool call begins
        let idx = text.find("<tool_call>").unwrap(); // Safe because has_tool_markers checked
        let normal_text = text[..idx].to_string();

        // Extract tool calls
        let mut tools = Vec::new();
        for captures in self.extractor.captures_iter(text) {
            if let Some(content_str) = captures.get(1) {
                let content = content_str.as_str().trim();
                
                // Detect format and parse accordingly
                match self.detect_format(content) {
                    ToolCallFormat::Json => {
                        // Try JSON format first
                        let parsed = serde_json::from_str::<Value>(content)
                            .map_err(|e| ParserError::ParsingFailed(e.to_string()))
                            .and_then(|v| self.parse_single_object(&v));

                        match parsed {
                            Ok(Some(tool)) => tools.push(tool),
                            Ok(None) => continue,
                            Err(e) => {
                                tracing::warn!("Failed to parse JSON tool call: {:?}", e);
                                continue;
                            }
                        }
                    }
                    ToolCallFormat::Xml => {
                        // Try XML format
                        match self.parse_xml_format(content) {
                            Ok(Some(tool)) => tools.push(tool),
                            Ok(None) => continue,
                            Err(e) => {
                                tracing::warn!("Failed to parse XML tool call: {:?}", e);
                                continue;
                            }
                        }
                    }
                    ToolCallFormat::Unknown => {
                        // Try both formats as fallback
                        let mut parsed = false;
                        
                        // Try JSON first
                        if let Ok(Some(tool)) = serde_json::from_str::<Value>(content)
                            .map_err(|e| ParserError::ParsingFailed(e.to_string()))
                            .and_then(|v| self.parse_single_object(&v))
                        {
                            tools.push(tool);
                            parsed = true;
                        }
                        
                        // Try XML if JSON failed
                        if !parsed {
                            if let Ok(Some(tool)) = self.parse_xml_format(content) {
                                tools.push(tool);
                            }
                        }
                    }
                }
            }
        }

        // If no tools were successfully parsed despite having markers, return entire text as fallback
        if tools.is_empty() {
            return Ok((text.to_string(), vec![]));
        }

        Ok((normal_text, tools))
    }

    async fn parse_incremental(
        &mut self,
        chunk: &str,
        tools: &[Tool],
    ) -> ParserResult<StreamingParseResult> {
        // Append new text to buffer
        self.buffer.push_str(chunk);
        let current_text = &self.buffer.clone();

        // Check if current_text has tool_call
        let has_tool_start = self.has_tool_markers(current_text)
            || (self.current_tool_id > 0 && current_text.starts_with(self.tool_call_separator))
            || self.in_xml_tool_call;

        if !has_tool_start {
            // Only clear buffer if we're sure no tool call is starting
            if helpers::ends_with_partial_token(&self.buffer, self.individual_tool_start_token)
                .is_none()
            {
                let normal_text = self.buffer.clone();
                self.buffer.clear();

                return Ok(StreamingParseResult {
                    normal_text,
                    calls: vec![],
                });
            } else {
                // Might be partial individual_tool_start_token, keep buffering
                return Ok(StreamingParseResult::default());
            }
        }

        // Build tool indices
        let tool_indices = helpers::get_tool_indices(tools);

        // Detect format: only check once, then remember it
        // Both JSON and XML formats use the same <tool_call> tags, difference is internal content
        // - JSON: <tool_call>\n{"name": "...", "arguments": {...}}\n</tool_call>
        // - XML: <tool_call>\n<function=name>\n<parameter=key>value</parameter>\n</function>\n</tool_call>
        let is_xml_format = if self.in_xml_tool_call {
            // Already in XML mode, continue with XML parser
            true
        } else if let Some(is_xml) = self.format_detected {
            // Format already detected, reuse the result (performance optimization)
            is_xml
        } else {
            // First time detection: check content inside <tool_call> tags
            // Only check if we have <tool_call> markers
            let detected = if self.has_tool_markers(current_text) {
                // Find content after <tool_call> tag
                if let Some(tool_call_pos) = current_text.find("<tool_call>") {
                    let after_tool_call = &current_text[tool_call_pos + "<tool_call>".len()..];
                    let trimmed = after_tool_call.trim();
                    
                    // XML format: has <function= tag (parameter= is optional, may come later)
                    if trimmed.contains("<function=") {
                        true
                    } else if trimmed.starts_with('{') {
                        // JSON format: starts with { (JSON object)
                        false
                    } else {
                        // Incomplete: haven't seen <function= or { yet
                        // Don't set format_detected yet, wait for more content
                        // This allows XML format to be detected when <function= appears later
                        return Ok(StreamingParseResult::default());
                    }
                } else {
                    // No <tool_call> found, default to JSON
                    false
                }
            } else {
                // No tool markers, not a tool call, default to JSON
                false
            };
            self.format_detected = Some(detected);
            detected
        };

        let mut result = if is_xml_format {
            // XML format streaming parsing
            self.parse_xml_incremental(current_text, &tool_indices)?
        } else {
            // JSON format streaming parsing
            // Determine start index for JSON parsing
            let start_idx = if let Some(pos) = current_text.find(self.individual_tool_start_token) {
                pos + self.individual_tool_start_token.len()
            } else if self.current_tool_id > 0 && current_text.starts_with(self.tool_call_separator) {
                self.tool_call_separator.len()
            } else {
                0
            };

            helpers::handle_json_tool_streaming(
                current_text,
                start_idx,
                &mut self.partial_json,
                &tool_indices,
                &mut self.buffer,
                &mut self.current_tool_id,
                &mut self.current_tool_name_sent,
                &mut self.streamed_args_for_tool,
                &mut self.prev_tool_call_arr,
            )?
        };

        // Qwen-specific: Handle partial end tokens in normal text
        // After tool calls complete, normal text might contain partial "</tool_call>" tags
        if !result.normal_text.is_empty() {
            self.normal_text_buffer.push_str(&result.normal_text);

            // Check if buffer contains complete end token (without leading newline)
            let end_token_without_newline = &self.individual_tool_end_token[1..]; // "</tool_call>"
            if self.normal_text_buffer.contains(end_token_without_newline) {
                // Complete end token found - clean it and return
                let cleaned_text = self
                    .normal_text_buffer
                    .replace(end_token_without_newline, "");
                self.normal_text_buffer.clear();
                result.normal_text = cleaned_text;
            } else {
                // Check if buffer might contain partial end token at the end
                if let Some(partial_match_len) = helpers::ends_with_partial_token(
                    &self.normal_text_buffer,
                    end_token_without_newline,
                ) {
                    // Keep potential partial match in buffer, return the rest
                    let split_point = self.normal_text_buffer.len() - partial_match_len;
                    result.normal_text = self.normal_text_buffer[..split_point].to_string();
                    self.normal_text_buffer = self.normal_text_buffer[split_point..].to_string();
                } else {
                    // No partial match, return all buffered text
                    result.normal_text = self.normal_text_buffer.clone();
                    self.normal_text_buffer.clear();
                }
            }
        }

        Ok(result)
    }

    fn has_tool_markers(&self, text: &str) -> bool {
        text.contains("<tool_call>")
    }

    fn get_unstreamed_tool_args(&self) -> Option<Vec<crate::tool_parser::types::ToolCallItem>> {
        helpers::get_unstreamed_args(&self.prev_tool_call_arr, &self.streamed_args_for_tool)
    }

    fn reset(&mut self) {
        helpers::reset_parser_state(
            &mut self.buffer,
            &mut self.prev_tool_call_arr,
            &mut self.current_tool_id,
            &mut self.current_tool_name_sent,
            &mut self.streamed_args_for_tool,
        );
        // Reset XML format state
        self.in_xml_tool_call = false;
        self.format_detected = None; // Reset format detection for next tool call
        self.xml_current_function_name.clear();
        self.xml_current_parameters.clear();
        self.xml_streamed_parameters.clear();
        self.in_parameter = false;
        self.current_parameter_key.clear();
        self.current_parameter_value.clear();
    }
}

impl QwenParser {
    /// Parse XML format tool calls incrementally (similar to Python Qwen3CoderDetector)
    fn parse_xml_incremental(
        &mut self,
        current_text: &str,
        tool_indices: &std::collections::HashMap<String, usize>,
    ) -> ParserResult<StreamingParseResult> {
        use crate::tool_parser::types::ToolCallItem;
        use regex::Regex;

        let mut normal_text = String::new();
        let mut calls: Vec<ToolCallItem> = vec![];

        // If we're not in a tool call and don't see a start token, return normal text
        if !self.in_xml_tool_call && !current_text.contains("<tool_call>") {
            normal_text = self.buffer.clone();
            self.buffer.clear();
            return Ok(StreamingParseResult {
                normal_text,
                calls,
            });
        }

        // Look for tool call start
        if !self.in_xml_tool_call {
            if let Some(s) = current_text.find("<tool_call>") {
                normal_text.push_str(&current_text[..s]);
                self.buffer = current_text[s + "<tool_call>".len()..].to_string();
                self.in_xml_tool_call = true;
                self.format_detected = Some(true); // Mark as XML format
                self.xml_current_function_name.clear();
                self.xml_current_parameters.clear();
                self.xml_streamed_parameters.clear();
                self.current_tool_name_sent = false;
                self.in_parameter = false;
                self.current_parameter_key.clear();
                self.current_parameter_value.clear();
            } else {
                // Partial start token, keep buffering
                return Ok(StreamingParseResult::default());
            }
        }

        // We're in a tool call, try to parse function name if not sent yet
        if !self.current_tool_name_sent {
            let function_pattern = Regex::new(r"<function=([^>]+)>")
                .map_err(|e| ParserError::ParsingFailed(format!("Invalid regex: {}", e)))?;
            
            if let Some(captures) = function_pattern.captures(&self.buffer) {
                if let Some(name_match) = captures.get(1) {
                    let function_name = name_match.as_str().trim().to_string();
                    
                    // Validate function name
                    if tool_indices.contains_key(&function_name) {
                        self.xml_current_function_name = function_name.clone();
                        self.current_tool_name_sent = true;
                        
                        // Initialize tool call tracking
                        if self.current_tool_id == -1 {
                            self.current_tool_id = 0;
                        }
                        
                        // Ensure tracking arrays are large enough
                        while self.prev_tool_call_arr.len() <= self.current_tool_id as usize {
                            self.prev_tool_call_arr.push(Value::Object(serde_json::Map::new()));
                        }
                        while self.streamed_args_for_tool.len() <= self.current_tool_id as usize {
                            self.streamed_args_for_tool.push(String::new());
                        }
                        
                        // Store tool call info
                        let mut tool_obj = serde_json::Map::new();
                        tool_obj.insert("name".to_string(), Value::String(function_name.clone()));
                        tool_obj.insert("arguments".to_string(), Value::Object(serde_json::Map::new()));
                        self.prev_tool_call_arr[self.current_tool_id as usize] = Value::Object(tool_obj);
                        
                        // Send tool name with empty parameters
                        calls.push(ToolCallItem {
                            tool_index: self.current_tool_id as usize,
                            name: Some(function_name),
                            parameters: String::new(),
                        });
                        
                        // Remove the processed function declaration
                        self.buffer = self.buffer[captures.get(0).unwrap().end()..].to_string();
                    } else {
                        // Invalid function name, reset state
                        self.in_xml_tool_call = false;
                        normal_text.push_str(&self.buffer);
                        self.buffer.clear();
                        return Ok(StreamingParseResult {
                            normal_text,
                            calls,
                        });
                    }
                }
            }
        }

        // Parse parameters incrementally
        if self.current_tool_name_sent {
            let param_start_pattern = Regex::new(r"<parameter=([^>]+)>")
                .map_err(|e| ParserError::ParsingFailed(format!("Invalid regex: {}", e)))?;
            
            // Check if we're entering a new parameter
            if !self.in_parameter {
                if let Some(cap) = param_start_pattern.captures(&self.buffer) {
                    if let Some(key_match) = cap.get(1) {
                        self.current_parameter_key = key_match.as_str().trim().to_string();
                        self.current_parameter_value.clear();
                        self.in_parameter = true;
                        
                        // Remove the opening tag from buffer
                        if let Some(m) = cap.get(0) {
                            self.buffer = self.buffer[m.end()..].to_string();
                        }
                    }
                }
            }
            
            // If we're in a parameter, accumulate value until we see </parameter>
            if self.in_parameter {
                if let Some(end_pos) = self.buffer.find("</parameter>") {
                    // Found complete parameter
                    let value = self.buffer[..end_pos].trim().to_string();
                    self.current_parameter_value.push_str(&value);
                    
                    // Remove the closing tag and processed content from buffer
                    self.buffer = self.buffer[end_pos + "</parameter>".len()..].to_string();
                    
                    // Parse and add the parameter
                    let key = self.current_parameter_key.clone();
                    let value_str = self.current_parameter_value.trim().to_string();
                    
                    // Try to parse value as JSON, otherwise use as string
                    let json_value = match serde_json::from_str::<Value>(&value_str) {
                        Ok(v) => v,
                        Err(_) => Value::String(value_str),
                    };
                    
                    // Add to current parameters
                    self.xml_current_parameters.insert(key.clone(), json_value.clone());
                    
                    // Stream the parameter update
                    let value_json = serde_json::to_string(&json_value)
                        .map_err(|e| ParserError::ParsingFailed(e.to_string()))?;
                    
                    let json_fragment = if self.xml_streamed_parameters.is_empty() {
                        format!("{{\"{}\": {}}}", key, value_json)
                    } else {
                        format!(", \"{}\": {}", key, value_json)
                    };
                    
                    calls.push(ToolCallItem {
                        tool_index: self.current_tool_id as usize,
                        name: None,
                        parameters: json_fragment.clone(),
                    });
                    
                    // Update streamed args
                    let current_args = &mut self.streamed_args_for_tool[self.current_tool_id as usize];
                    if current_args.is_empty() {
                        *current_args = format!("{{\"{}\": {}}}", key, value_json);
                    } else {
                        // Remove the closing brace, add new parameter, add closing brace
                        if current_args.ends_with('}') {
                            *current_args = format!("{}{}}}", &current_args[..current_args.len()-1], json_fragment);
                        } else {
                            *current_args = format!("{}{}", current_args, json_fragment);
                        }
                    }
                    
                    // Update streamed parameters
                    self.xml_streamed_parameters.insert(key, json_value);
                    
                    // Reset parameter state
                    self.in_parameter = false;
                    self.current_parameter_key.clear();
                    self.current_parameter_value.clear();
                    
                    // Update prev_tool_call_arr
                    if let Some(tool_obj) = self.prev_tool_call_arr[self.current_tool_id as usize].as_object_mut() {
                        tool_obj.insert("arguments".to_string(), Value::Object(self.xml_current_parameters.clone()));
                    }
                } else {
                    // Parameter value is incomplete, accumulate it
                    // Check if there's any content before a potential partial closing tag
                    if let Some(partial_end) = self.buffer.find("</") {
                        // There might be a partial closing tag, only take content before it
                        self.current_parameter_value.push_str(&self.buffer[..partial_end]);
                        self.buffer = self.buffer[partial_end..].to_string();
                    } else {
                        // No closing tag yet, accumulate all content
                        self.current_parameter_value.push_str(&self.buffer);
                        self.buffer.clear();
                    }
                }
            }
            
            // Check if tool call is complete
            if self.buffer.contains("</tool_call>") {
                // Complete the tool call
                if let Some(end_pos) = self.buffer.find("</tool_call>") {
                    self.buffer = self.buffer[end_pos + "</tool_call>".len()..].to_string();
                }
                self.in_xml_tool_call = false;
                self.format_detected = None; // Reset for next tool call
                self.current_tool_id += 1;
                self.xml_current_function_name.clear();
                self.xml_current_parameters.clear();
                self.xml_streamed_parameters.clear();
                self.current_tool_name_sent = false;
                self.in_parameter = false;
                self.current_parameter_key.clear();
                self.current_parameter_value.clear();
            }
        }

        Ok(StreamingParseResult {
            normal_text,
            calls,
        })
    }
}
