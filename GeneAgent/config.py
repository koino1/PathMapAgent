import os
from pathlib import Path

import openai
from openai import OpenAI


DEFAULT_MODEL = "gpt-4o"
DEFAULT_KEY_FILE = Path(__file__).resolve().parent / "openai_GeneAgent_key"
_CLIENT = None


def configure_openai():
    global _CLIENT
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # Prefer the local file alongside the GeneAgent code.  The current
        # working-directory lookup is retained for older project layouts.
        for key_path in (DEFAULT_KEY_FILE, Path.cwd() / DEFAULT_KEY_FILE.name):
            if key_path.exists():
                api_key = key_path.read_text(encoding="utf-8").strip()
                if api_key:
                    break
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set, and no key was found at "
            "GeneAgent/openai_GeneAgent_key."
        )
    openai.api_key = api_key
    _CLIENT = OpenAI(api_key=api_key)
    return api_key


def get_model_name():
    return os.getenv("OPENAI_MODEL", DEFAULT_MODEL)


def get_openai_client():
    global _CLIENT
    if _CLIENT is None:
        configure_openai()
    return _CLIENT


def message_to_dict(message):
    data = {
        "role": message.role,
        "content": message.content,
    }
    if getattr(message, "function_call", None):
        data["function_call"] = {
            "name": message.function_call.name,
            "arguments": message.function_call.arguments,
        }
    return data
