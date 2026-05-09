import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration for Football Society Agents
API_KEY = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY", "your_default_key")
BASE_URL = os.getenv("BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4-turbo-preview")

# App Constants
VERSION = "1.3.0-Cinderella"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
