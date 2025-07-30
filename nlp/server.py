from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os

# Add the current directory to Python path to import analyze_message
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from analyze_message import analyze_message
except ImportError:
    try:
        from test import analyze_message
    except ImportError:
        print("Error: Could not import analyze_message from analyze_message.py or test.py")
        sys.exit(1)

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from extension

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        message = data.get('message', '')
        
        if not message:
            return jsonify({"error": "No message provided"}), 400
        
        result = analyze_message(message)
        
        # Debug print to server console
        print(f"\n--- SERVER ANALYSIS ---")
        print(f"Message: {message}")
        print(f"Category: {result['category']}")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Reply: {result['reply']}")
        print(f"Instructions: {result['user_instructions']}")
        
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Server Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "Server is running"}), 200

if __name__ == '__main__':
    print("🚀 Starting BaitShift NLP Server...")
    print("📡 Extension will connect to: http://localhost:5000")
    app.run(host='localhost', port=5001, debug=True)
