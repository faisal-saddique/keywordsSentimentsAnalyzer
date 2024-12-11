import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
class Config:
    """Application configuration settings."""
    AHREFS_API_URL = "https://api.ahrefs.com/v3/serp-overview/serp-overview"
    OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
    AHREFS_AUTH_TOKEN = os.getenv("AHREFS_AUTH_TOKEN")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
