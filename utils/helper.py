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

        result = pipeline.invoke(query, config=config)
        print(f'Spent a total of {cb.total_tokens} tokens')

    return result


def get_chat_history(session_id: str, chat_map: dict ={}) -> InMemoryChatMessageHistory:
    if session_id not in chat_map:
        # if session ID doesn't exist, create a new chat history
        chat_map[session_id] = InMemoryChatMessageHistory()
    return chat_map[session_id]