import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from typing import Any, Dict, List
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult, ChatGeneration

class GeminiCallbackHandler(BaseCallbackHandler):
    """
    Callback Handler to track token usage by reading usage_metadata
    nested within the response Generation object.
    """
    def __init__(self):
        super().__init__()
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens_used = 0
        self.calls = 0

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Extracts usage_metadata from the nested message in the first generation."""
        if response.generations and response.generations[0]:
            first_generation = response.generations[0][0]
            
            if isinstance(first_generation, ChatGeneration) and hasattr(first_generation.message, 'usage_metadata'):
                usage_metadata = first_generation.message.usage_metadata
                
                if usage_metadata:
                    # Keys from the API are 'input_tokens' and 'output_tokens'
                    prompt_tokens = usage_metadata.get('input_tokens', 0)
                    completion_tokens = usage_metadata.get('output_tokens', 0)
                    # The API's total_tokens may include internal "thinking" tokens
                    total_tokens = usage_metadata.get('total_tokens', 0)

                    self.total_prompt_tokens += prompt_tokens
                    self.total_completion_tokens += completion_tokens
                    self.total_tokens_used += total_tokens

    def get_total_usage(self) -> Dict[str, int]:
        return {
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens_used": self.total_tokens_used,
            "total_llm_calls": self.calls
        }

    def reset(self) -> None:
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens_used = 0
        self.calls = 0