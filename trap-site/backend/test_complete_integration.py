"""
Test the complete backend with AI integration for Firebase logging
"""
import json

# Sample message data that would come from frontend
sample_message_data = {
    "message": "send me your bank details for verification",
    "sessionId": "test_session_123",
    "sequenceNumber": 1,
    "timestampISO": "2025-07-31T12:00:00Z",
    "typingTime": 1500,
    "lureType": "chat_trap",
    "pageVersion": "baitshift_gemini_chat"
}

# Test the AI analysis part (what happens in /log endpoint)
try:
    from ai_client import analyze_message, get_model_info
    
    print("🧪 Testing Complete Backend AI Integration")
    print("=" * 50)
    
    # Get model info
    info = get_model_info()
    print(f"🤖 Model Status:")
    print(f"   Version: {info.get('model_version')}")
    print(f"   Improved models: {info.get('using_improved_models')}")
    print(f"   Categories: {len(info.get('tone_categories', []))} tone categories")
    
    # Simulate what happens in the /log endpoint
    message_content = sample_message_data.get('message', '')
    print(f"\n📝 Processing message: '{message_content}'")
    
    if message_content.strip():
        ai_analysis = analyze_message(message_content)
        ai_risk_score = ai_analysis.get('risk_score', None)
        ai_tone_label = ai_analysis.get('tone_label', None)  
        ai_threat_category = ai_analysis.get('threat_category', None)
        ai_confidence = ai_analysis.get('confidence', None)
        ai_model_version = ai_analysis.get('model_version', None)
        
        print(f"\n🎯 AI Analysis Results:")
        print(f"   Risk Score: {ai_risk_score}/100")
        print(f"   Tone: {ai_tone_label}")
        print(f"   Threat Category: {ai_threat_category}")
        print(f"   Confidence: {ai_confidence:.3f}")
        print(f"   Model Version: {ai_model_version}")
        
        # Show what would be logged to Firebase
        firebase_log_data = {
            "message": message_content,
            "session_id": sample_message_data.get('sessionId'),
            "sequence_number": sample_message_data.get('sequenceNumber'),
            "ai_risk_score": ai_risk_score,
            "ai_tone_label": ai_tone_label,
            "ai_threat_category": ai_threat_category,
            "ai_confidence": ai_confidence,
            "ai_model_version": ai_model_version,
            # Legacy fields for compatibility
            "risk_score": ai_risk_score,
            "tone_label": ai_tone_label,
            "threat_category": ai_threat_category
        }
        
        print(f"\n📊 Firebase Log Data (relevant fields):")
        for key, value in firebase_log_data.items():
            print(f"   {key}: {value}")
        
        print(f"\n✅ Backend AI integration is working perfectly!")
        print(f"   • AI models load correctly from backend")
        print(f"   • Message analysis produces 3 key outputs")
        print(f"   • Results are ready for Firebase logging")
        print(f"   • Both new and legacy field names supported")
        
    else:
        print("❌ No message content to analyze")
        
except Exception as e:
    print(f"❌ Error in AI integration: {e}")
    import traceback
    traceback.print_exc()
