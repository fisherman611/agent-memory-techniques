import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.chat_history import InMemoryChatMessageHistory
from callbacks.manager import get_gemini_callback

def count_tokens(pipeline, query, config=None):
    with get_gemini_callback() as cb:

        if isinstance(query, str):
            query = {"query": query}

        if config is None:
            config = {"configurable": {"session_id": "default"}}

        # Explicitly pass the callback to LangChain
        if "callbacks" not in config:
            config["callbacks"] = [cb]
        elif isinstance(config["callbacks"], list):
            config["callbacks"].append(cb)
        else:
            config["callbacks"] = [config["callbacks"], cb]

        result = pipeline.invoke(query, config=config)
        usage = cb.get_total_usage()
        print(f"Spent a total of {usage['total_tokens_used']} tokens "
              f"(prompt: {usage['total_prompt_tokens']}, completion: {usage['total_completion_tokens']})")

    return result