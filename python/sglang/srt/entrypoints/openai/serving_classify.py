from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from fastapi import Request
from fastapi.responses import ORJSONResponse

from sglang.srt.entrypoints.openai.protocol import (
    ClassifyData,
    ClassifyRequest,
    ClassifyResponse,
    ClassifyUsage,
    ErrorResponse,
)
from sglang.srt.entrypoints.openai.serving_base import OpenAIServingBase
from sglang.srt.managers.io_struct import EmbeddingReqInput

if TYPE_CHECKING:
    from sglang.srt.managers.template_manager import TemplateManager
    from sglang.srt.managers.tokenizer_manager import TokenizerManager


class OpenAIServingClassify(OpenAIServingBase):
    """Handler for v1/classify requests"""

    def __init__(
        self,
        tokenizer_manager: TokenizerManager,
        template_manager: TemplateManager,
    ):
        super().__init__(tokenizer_manager)
        self.template_manager = template_manager

    def _request_id_prefix(self) -> str:
        return "classify-"

    def _convert_to_internal_request(
        self,
        request: ClassifyRequest,
        raw_request: Request = None,
    ) -> tuple[EmbeddingReqInput, ClassifyRequest]:
        """Convert classification request to internal embedding format"""
        # Convert to internal request format
        embedding_req = EmbeddingReqInput(
            text=request.input,
            rid=request.rid,
            priority=request.priority,
        )
        return embedding_req, request

    def _validate_request(self, request: ClassifyRequest) -> Optional[str]:
        """Validate that the input is not empty or whitespace only."""
        if not request.input or not request.input.strip():
            return "Input cannot be empty or whitespace only"
        return None

    def _get_id2label_mapping(self) -> Optional[Dict[int, str]]:
        """Get id2label mapping from model config."""
        try:
            # Try to get id2label from tokenizer_manager's model config
            if hasattr(self.tokenizer_manager, 'model_config') and self.tokenizer_manager.model_config:
                config = self.tokenizer_manager.model_config
                
                # Check for id2label in config
                if hasattr(config, 'id2label') and config.id2label:
                    return config.id2label
                
                # Check for num_labels and create default mapping if needed
                if hasattr(config, 'num_labels') and config.num_labels:
                    num_labels = config.num_labels
                    # Create default mapping: {0: "LABEL_0", 1: "LABEL_1", ...}
                    return {i: f"LABEL_{i}" for i in range(num_labels)}
            
            # Try to get from model_config directly if available
            if hasattr(self.tokenizer_manager, 'model_config') and hasattr(self.tokenizer_manager.model_config, 'id2label'):
                return self.tokenizer_manager.model_config.id2label
                
        except Exception as e:
            # Log the error but don't fail the request
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to get id2label mapping: {e}")
        
        return None

    async def _handle_non_streaming_request(
        self,
        adapted_request: EmbeddingReqInput,
        request: ClassifyRequest,
        raw_request: Request,
    ) -> Union[ClassifyResponse, ErrorResponse, ORJSONResponse]:
        """Handle non-streaming classification request."""
        # Generate request ID
        request_id = f"{self._request_id_prefix()}{uuid.uuid4().hex}"
        created_time = int(time.time())

        try:
            # Process the request using the existing classify endpoint
            # This uses the same backend as the /classify endpoint
            result = await self.tokenizer_manager.generate_request(
                adapted_request, raw_request
            ).__anext__()

            # Extract classification results from the response
            if hasattr(result, "data") and result.data:
                # Get id2label mapping from model config
                id2label = self._get_id2label_mapping()
                
                # Parse the classification results
                classify_data = []
                for i, item in enumerate(result.data):
                    if hasattr(item, "scores") and item.scores:
                        # Convert scores to probabilities using softmax
                        import torch
                        import torch.nn.functional as F
                        
                        scores = torch.tensor(item.scores, dtype=torch.float32)
                        probs = F.softmax(scores, dim=0).tolist()
                        
                        # Get the predicted class (highest probability)
                        predicted_class = torch.argmax(scores).item()
                        
                        # Use id2label mapping if available, otherwise fallback to Class_X
                        if id2label and predicted_class in id2label:
                            label = id2label[predicted_class]
                        else:
                            label = f"Class_{predicted_class}"
                        
                        classify_data.append(
                            ClassifyData(
                                index=i,
                                label=label,
                                probs=probs,
                                num_classes=len(probs),
                            )
                        )
                    else:
                        # Fallback: create a single class with probability 1.0
                        classify_data.append(
                            ClassifyData(
                                index=i,
                                label="Default",
                                probs=[1.0],
                                num_classes=1,
                            )
                        )

                # Create usage information
                usage = ClassifyUsage(
                    prompt_tokens=getattr(result, "prompt_tokens", 0),
                    total_tokens=getattr(result, "total_tokens", 0),
                    completion_tokens=0,
                    prompt_tokens_details=None,
                )

                # Create response
                response = ClassifyResponse(
                    id=request_id,
                    object="list",
                    created=created_time,
                    model=request.model,
                    data=classify_data,
                    usage=usage,
                )

                return ORJSONResponse(content=response.model_dump())

            else:
                # No classification data available, return error
                return ORJSONResponse(
                    content=ErrorResponse(
                        message="Classification model not available or failed to process",
                        type="server_error",
                        code=500,
                    ).model_dump(),
                    status_code=500,
                )

        except Exception as e:
            return ORJSONResponse(
                content=ErrorResponse(
                    message=f"Classification failed: {str(e)}",
                    type="server_error",
                    code=500,
                ).model_dump(),
                status_code=500,
            )

