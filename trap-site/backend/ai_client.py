"""
AI Client for BaitShift Backend
Handles communication with the AI model
"""
import sys
import os
from typing import Dict, Any, Optional

def setup_ai_path() -> None:
    """Add AI model directory to Python path"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    ai_model_path = os.path.join(current_dir, '..', 'ai_model')
    ai_model_path = os.path.abspath(ai_model_path)
    if ai_model_path not in sys.path:
        sys.path.insert(0, ai_model_path)

# Set up path and import AI functions
setup_ai_path()

try:
    # Import the AI functions (dynamic import - lint warnings are expected)
    from main import analyze_message as _analyze_message  # type: ignore
    from main import get_model_info as _get_model_info    # type: ignore
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    _analyze_message = None
    _get_model_info = None

def analyze_message(message: str) -> Dict[str, Any]:
    """Analyze message using AI models"""
    if not AI_AVAILABLE or _analyze_message is None:
        raise RuntimeError("AI models not available")
    return _analyze_message(message)

def get_model_info() -> Dict[str, Any]:
    """Get AI model information"""
    if not AI_AVAILABLE or _get_model_info is None:
        return {"error": "AI models not available"}
    return _get_model_info()

def is_ai_available() -> bool:
    """Check if AI models are available"""
    return AI_AVAILABLE
