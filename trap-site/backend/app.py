from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
import requests


import hashlib
import json
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore

# Import AI inference module
try:
    from ai_client import analyze_message, get_model_info, is_ai_available
    AI_AVAILABLE = is_ai_available()
    if AI_AVAILABLE:
        print("✅ AI inference module loaded successfully")
        # Print model info
        model_info = get_model_info()
        print(f"   Model version: {model_info.get('model_version')}")
        print(f"   Using improved models: {model_info.get('using_improved_models')}")
    else:
        print("⚠️ AI models not available")
except Exception as e:
    AI_AVAILABLE = False
    analyze_message = None
    get_model_info = None
    print(f"⚠️ AI inference not available: {e}")
    print("   Make sure the ai_model directory contains the required files")

try:
    from user_agents import parse
    USER_AGENTS_AVAILABLE = True
except ImportError:
    USER_AGENTS_AVAILABLE = False
    print("⚠️ user-agents not available - user agent parsing will be limited")

try:
    import geoip2.database
    GEOIP_AVAILABLE = True
except ImportError:
    GEOIP_AVAILABLE = False
    print("⚠️ geoip2 not available - geolocation will be limited")

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Load Gemini API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize Firebase
cred = credentials.Certificate("firebase-service-account.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# Initialize GeoIP reader (download GeoLite2-City.mmdb from MaxMind)
geoip_reader = None
if GEOIP_AVAILABLE:
    try:
        geoip_reader = geoip2.database.Reader('GeoLite2-City.mmdb')
    except:
        print("⚠️ GeoIP database not found - geolocation will be limited")

def hash_ip(ip_address):
    """Hash IP address for privacy"""
    return hashlib.sha256(ip_address.encode()).hexdigest()

def get_geolocation(ip_address):
    """Get geolocation from IP address using free API services"""
    
    # First try MaxMind database if available
    if GEOIP_AVAILABLE and geoip_reader:
        try:
            if ip_address != 'unknown' and not ip_address.startswith('127.') and not ip_address.startswith('192.168.'):
                response = geoip_reader.city(ip_address)
                result = {
                    "country": response.country.name,
                    "city": response.city.name,
                    "latitude": float(response.location.latitude) if response.location.latitude else None,
                    "longitude": float(response.location.longitude) if response.location.longitude else None,
                    "source": "maxmind"
                }
                return result
        except Exception as e:
            pass  # Fallback to API
    
    # Fallback to free IP geolocation API
    if ip_address and ip_address != 'unknown' and not ip_address.startswith('127.') and not ip_address.startswith('192.168.'):
        try:
            # Use ipapi.co free service (1000 requests/day)
            response = requests.get(f'https://ipapi.co/{ip_address}/json/', timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                
                result = {
                    "country": data.get('country_name', 'Unknown'),
                    "city": data.get('city', 'Unknown'),
                    "latitude": data.get('latitude'),
                    "longitude": data.get('longitude'),
                    "region": data.get('region', 'Unknown'),
                    "source": "ipapi"
                }
                return result
        except Exception as e:
            pass  # Return Unknown
    
    return {
        "country": "Unknown",
        "city": "Unknown", 
        "latitude": None,
        "longitude": None,
        "source": "none"
    }

def parse_user_agent(user_agent_string):
    """Parse user agent for device/browser info"""
    if not USER_AGENTS_AVAILABLE:
        return {
            "browser": "Unknown",
            "browser_version": "Unknown",
            "os": "Unknown", 
            "os_version": "Unknown",
            "device": "Unknown",
            "is_mobile": False,
            "is_bot": False
        }
    
    try:
        print(USER_AGENTS_AVAILABLE)
        user_agent = parse(user_agent_string)
        return {
            "browser": user_agent.browser.family,
            "browser_version": user_agent.browser.version_string,
            "os": user_agent.os.family,
            "os_version": user_agent.os.version_string,
            "device": user_agent.device.family,
            "is_mobile": user_agent.is_mobile,
            "is_bot": user_agent.is_bot
        }
    except:
        return {
            "browser": "Unknown",
            "browser_version": "Unknown",
            "os": "Unknown", 
            "os_version": "Unknown",
            "device": "Unknown",
            "is_mobile": False,
            "is_bot": False
        }

def detect_copy_paste(message, typing_time=None, message_length=None):
    """Detect potential copy-paste behavior"""
    if not typing_time or not message_length:
        return False
    
    # Rough heuristic: if typing time is too fast for message length
    # Average typing speed is ~40 WPM = ~200 chars/min = ~3.3 chars/sec
    expected_typing_time = message_length / 3.3  # seconds
    
    return typing_time < (expected_typing_time * 0.3)  # 30% of expected time

@app.route("/log", methods=["POST"])
def log_message():
    """Enhanced endpoint to log comprehensive attacker data"""
    try:
        data = request.get_json()
        client_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)
        
        # Parse user agent
        user_agent_string = request.headers.get('User-Agent', '')
        parsed_ua = parse_user_agent(user_agent_string)
        
        # Get geolocation - use provided clientIP from frontend if available
        ip_for_geolocation = data.get('clientIP', client_ip)
        geolocation = get_geolocation(ip_for_geolocation)
        
        # Calculate copy-paste indicator
        copy_paste_indicator = detect_copy_paste(
            data.get('message', ''),
            data.get('typingTime'),
            len(data.get('message', ''))
        )
        
        # AI ANALYSIS - GET 3 MODEL PARAMETERS
        message_content = data.get('message', '')
        if AI_AVAILABLE and message_content.strip():
            try:
                ai_analysis = analyze_message(message_content)
                ai_risk_score = ai_analysis.get('risk_score', None)
                ai_tone_label = ai_analysis.get('tone_label', None)  
                ai_threat_category = ai_analysis.get('threat_category', None)
                ai_confidence = ai_analysis.get('confidence', None)
                ai_model_version = ai_analysis.get('model_version', None)
                print(f"🤖 AI Analysis: Risk={ai_risk_score}, Tone={ai_tone_label}, Category={ai_threat_category}")
            except Exception as e:
                print(f"❌ AI analysis failed: {e}")
                ai_risk_score = None
                ai_tone_label = None
                ai_threat_category = None
                ai_confidence = None
                ai_model_version = None
        else:
            ai_risk_score = None
            ai_tone_label = None
            ai_threat_category = None
            ai_confidence = None
            ai_model_version = None
        
        # Comprehensive log data with all required fields
        log_data = {
            # Basic message data
            "timestamp": datetime.utcnow(),
            "timestamp_iso": data.get('timestampISO'),
            "message": data.get('message', ''),
            "message_length": len(data.get('message', '')),
            "message_hash": hashlib.sha256(data.get('message', '').encode()).hexdigest(),
            
            # Session tracking
            "session_id": data.get('sessionId'),
            "session_start": data.get('sessionStart'),
            "session_end": data.get('sessionEnd'),
            "sequence_number": data.get('sequenceNumber', 0),
            "delay_since_last_message": data.get('delaySinceLastMessage', 0),
            "num_messages": data.get('sequenceNumber', 0),  # Will be updated per session
            
            # Network & device data
            "ip_address_hashed": hash_ip(client_ip),
            "user_agent": user_agent_string,
            "user_agent_parsed": parsed_ua,
            
            # Geolocation
            "geolocation": geolocation,
            
            # Behavioral indicators
            "copy_paste_indicator": copy_paste_indicator,
            "typing_time": data.get('typingTime'),
            
            # Entry tracking
            "lure_type": data.get('lureType', 'chat_trap'),
            "click_path": data.get('clickPath', []),
            "entry_url": data.get('entryUrl'),
            
            # Page/version tracking
            "page_version": data.get('pageVersion', 'baitshift_gemini_chat'),
            
            # Timestamps
            "server_timestamp": datetime.utcnow(),
            "server_timestamp_firestore": firestore.SERVER_TIMESTAMP,
            
            # AI model outputs (BaitShift improved models)
            "ai_risk_score": ai_risk_score,
            "ai_tone_label": ai_tone_label,
            "ai_threat_category": ai_threat_category,
            "ai_confidence": ai_confidence,
            "ai_model_version": ai_model_version,
            
            # Legacy fields for backward compatibility
            "risk_score": ai_risk_score,  # Same as ai_risk_score
            "tone_label": ai_tone_label,  # Same as ai_tone_label
            "threat_category": ai_threat_category,  # Same as ai_threat_category
        }
        
        # Log to Firebase trap_logs collection
        doc_ref = db.collection("trap_logs").add(log_data)
        print(f"✅ Enhanced log saved: Seq {data.get('sequenceNumber', '?')} | IP: {client_ip[:8]}...")
        
        return jsonify({"success": True, "doc_id": doc_ref[1].id})
        
    except Exception as e:
        print(f"❌ Error logging to Firebase: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/session", methods=["POST"])
def update_session():
    """Update session data (start/end times, total messages)"""
    try:
        data = request.get_json()
        session_id = data.get('sessionId')
        
        session_data = {
            "session_id": session_id,
            "session_start": data.get('sessionStart'),
            "session_end": data.get('sessionEnd'),
            "total_messages": data.get('totalMessages', 0),
            "session_duration": data.get('sessionDuration', 0),
            "click_path": data.get('clickPath', []),
            "exit_action": data.get('exitAction', 'unknown'),
            "server_timestamp": datetime.utcnow()
        }
        
        # Save session summary
        db.collection("trap_sessions").document(session_id).set(session_data)
        print(f"✅ Session updated: {session_id}")
        
        return jsonify({"success": True})
        
    except Exception as e:
        print(f"❌ Error updating session: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    message = data.get("message", "")
    instruction = data.get("instruction", "")

    prompt = f"{instruction}\n\nUser: {message}\nAI:"
    headers = { "Content-Type": "application/json" }

    body = {
        "contents": [ { "parts": [ { "text": prompt } ] } ],
        "generationConfig": { "temperature": 0.7, "maxOutputTokens": 60 }
    }

    try:
        response = requests.post(
            "https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent",
            headers=headers,
            params={ "key": GEMINI_API_KEY },
            json=body
        )

        if response.status_code == 200:
            reply = response.json()["candidates"][0]["content"]["parts"][0]["text"]
        else:
            reply = "hmm idk what to say 🤔"

    except Exception as e:
        print("Gemini API error:", e)
        reply = "sorry i'm being weird today lol 😅"

    return jsonify({ "reply": reply })

if __name__ == "__main__":
    app.run(port=5000, debug=True)
