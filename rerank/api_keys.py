import os
from typing import Dict

from dotenv import load_dotenv


def get_openai_api_key(resource='baidu', model_name='gpt-4o-2024-08-06') -> str:
    load_dotenv(dotenv_path=f".env.local")
    key = os.getenv("OPEN_AI_API_KEY")
    return key

