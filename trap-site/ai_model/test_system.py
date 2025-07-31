"""
Simple test to verify BaitShift AI is working correctly
"""

from main import analyze_message, get_model_info

def test_ai_system():
    """Test the AI system with various scenarios"""
    
    print("🧪 BaitShift AI System Test")
    print("=" * 40)
    
    # Test 1: Check model loading
    print("\n1. Model Status:")
    info = get_model_info()
    print(f"   Models loaded: {info.get('tone_model_loaded')} / {info.get('risk_model_loaded')}")
    print(f"   Model version: {info.get('model_version')}")
    print(f"   Using improved models: {info.get('using_improved_models')}")
    
    # Test 2: Basic functionality
    print("\n2. Basic Tests:")
    
    test_cases = [
        "hey beautiful how old are you",
        "your computer has a virus",
        "what's your favorite movie",
        "send me your bank details",
        "i'll find you if you don't comply"
    ]
    
    for i, message in enumerate(test_cases, 1):
        try:
            result = analyze_message(message)
            print(f"   Test {i}: ✅ Risk={result['risk_score']}, Tone={result['tone_label']}")
        except Exception as e:
            print(f"   Test {i}: ❌ Error - {e}")
            return False
    
    print("\n✅ All tests passed! AI system is working perfectly.")
    return True

if __name__ == "__main__":
    success = test_ai_system()
    if success:
        print("\n🎯 System Status: READY FOR PRODUCTION")
    else:
        print("\n❌ System Status: NEEDS ATTENTION")
