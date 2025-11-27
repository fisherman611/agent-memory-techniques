from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from callbacks.gemini_info import GeminiCallbackHandler
from contextlib import contextmanager
from contextvars import ContextVar

gemini_callback_var: ContextVar[Optional[GeminiCallbackHandler]] = ContextVar(
    "gemini_callback", default=None
)

@contextmanager
def get_gemini_callback():
    cb = GeminiCallbackHandler()
    gemini_callback_var.set(cb)
    yield cb
    gemini_callback_var.set(None)