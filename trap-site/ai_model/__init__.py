"""
BaitShift AI Model Package
Improved AI inference for scammer detection
"""

from .main import analyze_message, get_model_info

__version__ = "2.0.0"
__author__ = "BaitShift Team"

# Expose main functions at package level
__all__ = ['analyze_message', 'get_model_info']
