"""
Test script for trained AI models
Run this after training to verify models work correctly
"""

import sys
import os

# Add ai_model directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_inference import TrapSiteAI

def test_models():
    """Test the trained AI models with various message types"""
    
    # Initialize AI
    ai = TrapSiteAI(models_dir='models')
    
    # Test messages covering different categories
    test_cases = [
        # Romance scam messages
        {
            "message": "hey beautiful how old are you?",
            "expected_category": "Romance_Scam",
            "expected_tone": "Manipulative",
            "expected_risk_range": (70, 90)
        },
        {
            "message": "you're so mature for your age baby",
            "expected_category": "Romance_Scam", 
            "expected_tone": "Manipulative",
            "expected_risk_range": (80, 95)
        },
        
        # Tech support scams
        {
            "message": "your computer has been infected call this number",
            "expected_category": "Tech_Support_Scam",
            "expected_tone": "Urgent",
            "expected_risk_range": (85, 100)
        },
        {
            "message": "microsoft detected virus download this tool",
            "expected_category": "Tech_Support_Scam",
            "expected_tone": "Technical",
            "expected_risk_range": (80, 95)
        },
        
        # Phishing attempts
        {
            "message": "click this link to verify your account",
            "expected_category": "Phishing",
            "expected_tone": "Urgent",
            "expected_risk_range": (85, 100)
        },
        
        # Financial scams
        {
            "message": "send me your bank details for money transfer",
            "expected_category": "Financial_Scam",
            "expected_tone": "Technical",
            "expected_risk_range": (85, 100)
        },
        
        # Threatening messages
        {
            "message": "if you don't comply I'll find you",
            "expected_category": "Threat",
            "expected_tone": "Threatening",
            "expected_risk_range": (90, 100)
        },
        
        # Information gathering
        {
            "message": "what school do you go to",
            "expected_category": "Information_Gathering",
            "expected_tone": "Friendly",
            "expected_risk_range": (50, 70)
        },
        
        # Low risk messages
        {
            "message": "what's your favorite movie",
            "expected_category": "Low_Risk_Social",
            "expected_tone": "Friendly",
            "expected_risk_range": (0, 30)
        },
        {
            "message": "how was your day",
            "expected_category": "Low_Risk_Social",
            "expected_tone": "Friendly",
            "expected_risk_range": (0, 30)
        }
    ]
    
    print("🧪 Testing Trained AI Models")
    print("=" * 60)
    
    # Get model info
    model_info = ai.get_model_info()
    print("🤖 Model Status:")
    for key, value in model_info.items():
        print(f"   {key}: {value}")
    
    if not model_info["tone_model_loaded"] or not model_info["risk_model_loaded"]:
        print("\n❌ Models not fully loaded. Please run train_model.py first.")
        return False
    
    print(f"\n🎯 Running {len(test_cases)} test cases...")
    print("=" * 60)
    
    passed_tests = 0
    failed_tests = 0
    
    for i, test_case in enumerate(test_cases, 1):
        message = test_case["message"]
        result = ai.analyze_message(message)
        
        print(f"\nTest {i}: {message}")
        print(f"Results:")
        print(f"   Risk Score: {result['risk_score']}/100")
        print(f"   Tone: {result['tone_label']} (confidence: {result['confidence']:.2f})")
        print(f"   Category: {result['threat_category']}")
        
        # Check predictions against expectations
        risk_in_range = test_case["expected_risk_range"][0] <= result['risk_score'] <= test_case["expected_risk_range"][1]
        
        print(f"Expected:")
        print(f"   Risk Range: {test_case['expected_risk_range'][0]}-{test_case['expected_risk_range'][1]} {'✅' if risk_in_range else '❌'}")
        print(f"   Category: {test_case['expected_category']} {'✅' if result['threat_category'] == test_case['expected_category'] else '❌'}")
        print(f"   Tone: {test_case['expected_tone']} {'✅' if result['tone_label'] == test_case['expected_tone'] else '❌'}")
        
        # Count pass/fail
        if (risk_in_range and 
            result['threat_category'] == test_case['expected_category'] and 
            result['tone_label'] == test_case['expected_tone']):
            passed_tests += 1
            print("Status: ✅ PASSED")
        else:
            failed_tests += 1
            print("Status: ❌ FAILED")
    
    print(f"\n🎯 Test Results Summary:")
    print("=" * 40)
    print(f"✅ Passed: {passed_tests}/{len(test_cases)}")
    print(f"❌ Failed: {failed_tests}/{len(test_cases)}")
    print(f"Success Rate: {(passed_tests/len(test_cases)*100):.1f}%")
    
    if passed_tests >= len(test_cases) * 0.8:  # 80% pass rate
        print("\n🎉 Models are working well! Ready for production.")
        return True
    else:
        print("\n⚠️ Models may need additional training or tuning.")
        return False

def test_edge_cases():
    """Test edge cases and unusual inputs"""
    print("\n🔍 Testing Edge Cases...")
    print("=" * 40)
    
    ai = TrapSiteAI(models_dir='models')
    
    edge_cases = [
        "",  # Empty message
        "a",  # Very short message
        "hello" * 100,  # Very long message
        "123 !@# $%^",  # Numbers and symbols
        "hEy BeAuTiFuL",  # Mixed case
        "ur soooo cute lol omg",  # Internet slang
    ]
    
    for message in edge_cases:
        try:
            result = ai.analyze_message(message)
            print(f"Input: '{message[:50]}{'...' if len(message) > 50 else ''}'")
            print(f"Output: Risk={result['risk_score']}, Tone={result['tone_label']}, Category={result['threat_category']}")
            print("✅ Handled successfully")
        except Exception as e:
            print(f"Input: '{message[:50]}{'...' if len(message) > 50 else ''}'")
            print(f"❌ Error: {e}")
        print()

if __name__ == "__main__":
    try:
        success = test_models()
        test_edge_cases()
        
        if success:
            print("\n🚀 All tests passed! Models ready for integration.")
        else:
            print("\n⚠️ Some tests failed. Check model training.")
            
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        print("Make sure models are trained and all dependencies are installed.")
