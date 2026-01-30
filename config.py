"""
Configuration for VAPI-Exotel Bridge
"""
import os
from dotenv import load_dotenv

load_dotenv()

# VAPI Configuration
VAPI_API_KEY = os.getenv("VAPI_API_KEY")
VAPI_ASSISTANT_ID = os.getenv("VAPI_ASSISTANT_ID")
VAPI_API_URL = "https://api.vapi.ai"

# Bridge Server Configuration
BRIDGE_HOST = os.getenv("BRIDGE_HOST", "0.0.0.0")
BRIDGE_PORT = int(os.getenv("BRIDGE_PORT", "8080"))

# n8n Webhook (optional - for sending call results)
N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL", "")

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Validate required config
if not VAPI_API_KEY:
    raise ValueError("VAPI_API_KEY is required in .env file")
if not VAPI_ASSISTANT_ID:
    raise ValueError("VAPI_ASSISTANT_ID is required in .env file")
