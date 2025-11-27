import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from typing import Any, Dict, List
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult

class GeminiCallbackHandler(BaseCallbackHandler):
    """Callback handler that tracks Gemini token usage."""

    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    successful_requests: int = 0
    total_cost: float = 0.0   # (Optional — you can compute cost per model)

    def __repr__(self) -> str:
        return (
            f"Tokens Used: {self.total_tokens}\n"
            f"\tPrompt Tokens: {self.prompt_tokens}\n"
            f"\tCompletion Tokens: {self.completion_tokens}\n"
            f"Successful Requests: {self.successful_requests}\n"
            f"Total Cost (USD): ${self.total_cost}"
        )

    @property
    def always_verbose(self) -> bool:
        return True

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Called when LLM starts — no-op for now."""
        pass

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Streaming token hook — not used for Gemini yet."""
        pass

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Collect Gemini token usage."""
        self.successful_requests += 1

        # In Gemini, token usage is stored in response.generations[0][0].generation_info
        # BUT LangChain normalizes it to response.llm_output["usage_metadata"]
        usage = None

        if response.llm_output:
            usage = response.llm_output.get("usage_metadata")

        if not usage:
            return None

        # Gemini fields:
        # - prompt_token_count
        # - candidates_token_count
        # - total_token_count
        prompt_toks = usage.get("prompt_token_count", 0)
        completion_toks = usage.get("candidates_token_count", 0)
        total_toks = usage.get("total_token_count", prompt_toks + completion_toks)

        self.prompt_tokens += prompt_toks
        self.completion_tokens += completion_toks
        self.total_tokens += total_toks

        # (Optional) cost tracking — fill later if needed
        # self.total_cost += compute_gemini_cost(model_name, prompt_toks, completion_toks)

    def __copy__(self):
        return self

    def __deepcopy__(self, memo: Any):
        return self