"""
Configuration settings for the SWIFT processing system
"""

import os
from pathlib import Path
from typing import Dict, Any

from dotenv import load_dotenv

# Load secrets from project/.env (resolved relative to this file, so it works
# from any working directory). Variables already set in the shell take precedence.
load_dotenv(Path(__file__).resolve().parent / ".env")


class Config:
    """Configuration class for the SWIFT processing system"""

    # System settings
    MESSAGE_COUNT = 10
    BANK_COUNT = 5

    # Processing settings
    MAX_WORKERS = 8
    BATCH_SIZE = 50

    # OpenAI settings (read by services/llm_service.py; the key comes from .env)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

    @classmethod
    def get_all_settings(cls) -> Dict[str, Any]:
        """Get all configuration settings as a dictionary"""
        return {
            attr: getattr(cls, attr)
            for attr in dir(cls)
            if not attr.startswith('_') and not callable(getattr(cls, attr))
        }
