from dotenv import load_dotenv
from src.simulation.runtime import environment_snapshot

# Load environment variables from .env file
load_dotenv()

# Configuration for Football Society Agents
_ENV = environment_snapshot()
API_KEY = (
    _ENV.get("DEEPSEEK_API_KEY")
    or _ENV.get("API_KEY")
    or _ENV.get("OPENAI_API_KEY", "your_default_key")
)
BASE_URL = _ENV.get("BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = _ENV.get("MODEL_NAME", "gpt-4-turbo-preview")

# App Constants
VERSION = "1.3.0-Cinderella"
LOG_LEVEL = _ENV.get("LOG_LEVEL", "INFO")
