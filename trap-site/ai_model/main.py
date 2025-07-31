"""
AI inference module for BaitShift trap site
UPDATED: Now uses the improved models with better performance
"""

# Import the improved AI inference
from core_ai import ImprovedTrapSiteAI

# Use the improved AI as the main instance
trap_ai = ImprovedTrapSiteAI()

# For backward compatibility, expose the main functions
def analyze_message(message):
    """Analyze message using improved AI models"""
    return trap_ai.analyze_message(message)

def get_model_info():
    """Get improved model information"""
    return trap_ai.get_model_info()

# Legacy functions for backward compatibility
def predict_tone(message):
    """Predict tone using improved model"""
    return trap_ai.predict_tone(message)

def predict_risk_score(message):
    """Predict risk score using improved ensemble model"""
    return trap_ai.predict_risk_score(message)

if __name__ == "__main__":
    print("🎯 BaitShift AI - Using Improved Models")
    print("=" * 50)
    
    # Print model info
    info = get_model_info()
    print("🤖 AI Model Status:")
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    # Run test
    test_messages = [
        "hey beautiful how old are you?",
        "your computer has been infected call us now", 
        "what's your favorite movie?",
        "send me your bank details for verification",
        "if you don't comply I'll find you"
    ]
    
    print("\n🧪 Testing AI Performance:")
    print("=" * 40)
    
    for message in test_messages:
        result = analyze_message(message)
        print(f"\nMessage: '{message}'")
        print(f"Risk Score: {result['risk_score']}/100")
        print(f"Tone: {result['tone_label']} (confidence: {result['confidence']})")
        print(f"Category: {result['threat_category']}")
        print(f"Model: {result['model_version']}")
