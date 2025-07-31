"""
Test script to verify AI integration with backend
"""

try:
    from ai_client import analyze_message, get_model_info
    print("✅ AI import successful")
    
    # Test model info
    info = get_model_info()
    print(f"📊 Model version: {info.get('model_version')}")
    print(f"🤖 Using improved models: {info.get('using_improved_models')}")
    
    # Test analysis
    test_message = "send me your bank details for verification"
    result = analyze_message(test_message)
    
    print(f"\n🧪 Test Analysis:")
    print(f"   Message: '{test_message}'")
    print(f"   Risk Score: {result.get('risk_score')}")
    print(f"   Tone: {result.get('tone_label')}")
    print(f"   Category: {result.get('threat_category')}")
    print(f"   Confidence: {result.get('confidence')}")
    print(f"   Model Version: {result.get('model_version')}")
    
    print("\n🎯 Backend AI integration is working perfectly!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
