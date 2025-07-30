"""
AI inference module for BaitShift trap site
Provides real-time threat analysis for attacker messages
"""

import torch
import pickle
import numpy as np
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import os
import json
import re

def normalize_text(text):
    """Normalize text for consistent processing - matches training normalization"""
    text = str(text).lower()
    # Expand contractions
    contractions = {
        "you're": "you are",
        "we're": "we are", 
        "they're": "they are",
        "can't": "cannot",
        "won't": "will not",
        "don't": "do not",
        "i'm": "i am",
        "i'll": "i will",
        "i've": "i have",
        "what's": "what is",
        "that's": "that is",
        "there's": "there is",
        "it's": "it is"
    }
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

class TrapSiteAI:
    def __init__(self, models_dir='ai_model/models'):
        self.models_dir = models_dir
        self.tone_model = None
        self.tone_tokenizer = None
        self.tone_encoder = None
        self.risk_model = None
        self.risk_vectorizer = None
        self.metadata = None
        self.load_models()
    
    def load_models(self):
        """Load trained models for inference"""
        try:
            # Load metadata if available
            metadata_path = os.path.join(self.models_dir, 'training_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                print(f"✅ Loaded training metadata from {self.metadata['training_date']}")
            
            # Load tone classification model
            tone_model_path = os.path.join(self.models_dir, 'tone_model')
            tone_tokenizer_path = os.path.join(self.models_dir, 'tone_tokenizer')
            tone_encoder_path = os.path.join(self.models_dir, 'tone_encoder.pkl')
            
            if os.path.exists(tone_model_path) and os.path.exists(tone_encoder_path):
                self.tone_model = DistilBertForSequenceClassification.from_pretrained(tone_model_path)
                self.tone_tokenizer = DistilBertTokenizer.from_pretrained(tone_tokenizer_path)
                
                with open(tone_encoder_path, 'rb') as f:
                    self.tone_encoder = pickle.load(f)
                
                self.tone_model.eval()
                print("✅ Tone classification model loaded")
            else:
                print("⚠️ Tone classification model not found")
            
            # Load risk scoring model
            risk_model_path = os.path.join(self.models_dir, 'risk_model.pkl')
            risk_vectorizer_path = os.path.join(self.models_dir, 'risk_vectorizer.pkl')
            
            if os.path.exists(risk_model_path) and os.path.exists(risk_vectorizer_path):
                with open(risk_model_path, 'rb') as f:
                    self.risk_model = pickle.load(f)
                
                with open(risk_vectorizer_path, 'rb') as f:
                    self.risk_vectorizer = pickle.load(f)
                
                print("✅ Risk scoring model loaded")
            else:
                print("⚠️ Risk scoring model not found")
            
        except Exception as e:
            print(f"⚠️ Error loading AI models: {e}")
            print("🔄 Models will return default values until training is complete")
    
    def predict_tone(self, message):
        """Predict tone label for a message"""
        if not self.tone_model or not self.tone_tokenizer or not self.tone_encoder:
            return {"tone_label": "Unknown", "confidence": 0.0}
        
        try:
            # Normalize message text
            normalized_message = normalize_text(message)
            
            # Tokenize message
            inputs = self.tone_tokenizer(
                normalized_message,
                truncation=True,
                padding='max_length',
                max_length=128,
                return_tensors='pt'
            )
            
            # Get prediction
            with torch.no_grad():
                outputs = self.tone_model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(predictions, dim=-1).item()
                confidence = predictions[0][predicted_class].item()
            
            # Decode prediction
            tone_label = self.tone_encoder.inverse_transform([predicted_class])[0]
            
            return {
                "tone_label": tone_label,
                "confidence": round(confidence, 3)
            }
            
        except Exception as e:
            print(f"❌ Error predicting tone: {e}")
            return {"tone_label": "Unknown", "confidence": 0.0}
    
    def predict_risk_score(self, message):
        """Predict risk score for a message (0-100)"""
        if not self.risk_model or not self.risk_vectorizer:
            return 50  # Default medium risk
        
        try:
            # Normalize message text
            normalized_message = normalize_text(message)
            
            # Vectorize message
            message_tfidf = self.risk_vectorizer.transform([normalized_message])
            
            # Get prediction (0-1 range)
            risk_normalized = self.risk_model.predict(message_tfidf)[0]
            
            # Convert to 0-100 scale and clamp
            risk_score = max(0, min(100, int(risk_normalized * 100)))
            
            return risk_score
            
        except Exception as e:
            print(f"❌ Error predicting risk score: {e}")
            return 50
    
    def analyze_message(self, message):
        """Complete analysis of an attacker message"""
        if not message or not message.strip():
            return {
                "risk_score": 0,
                "tone_label": "Empty",
                "confidence": 1.0,
                "threat_category": "Empty_Message"
            }
        
        # Get predictions
        tone_result = self.predict_tone(message)
        risk_score = self.predict_risk_score(message)
        
        # Extract results
        tone_label = tone_result["tone_label"]
        confidence = tone_result["confidence"]
        
        return {
            "risk_score": risk_score,
            "tone_label": tone_label,
            "confidence": confidence,
            "threat_category": self._classify_threat_category(message, tone_label, risk_score)
        }
    
    def _classify_threat_category(self, message, tone_label, risk_score):
        """Rule-based threat categorization"""
        message_lower = message.lower()
        
        # High-confidence categorization based on keywords
        if any(word in message_lower for word in ['beautiful', 'cute', 'meet', 'private', 'age', 'old are you', 'send pic']):
            return "Romance_Scam"
        
        if any(word in message_lower for word in ['computer', 'virus', 'download', 'remote', 'tech', 'microsoft', 'infected']):
            return "Tech_Support_Scam"
        
        if any(word in message_lower for word in ['account', 'verify', 'click', 'link', 'suspended', 'login', 'password']):
            return "Phishing"
        
        if any(word in message_lower for word in ['money', 'bank', 'payment', 'invest', 'crypto', 'rich', 'dollars']):
            return "Financial_Scam"
        
        if any(word in message_lower for word in ['find you', 'know where', 'or else', 'hurt you', 'regret']):
            return "Threat"
        
        if any(word in message_lower for word in ['school', 'where do you live', 'what grade', 'home alone', 'parents']):
            return "Information_Gathering"
        
        # Fallback based on risk score and tone
        if risk_score >= 80:
            if tone_label == "Threatening":
                return "High_Risk_Threat"
            elif tone_label == "Urgent":
                return "High_Risk_Scam"
            else:
                return "High_Risk_Unknown"
        elif risk_score >= 50:
            return "Medium_Risk_Unknown"
        else:
            return "Low_Risk_Social"
    
    def get_model_info(self):
        """Get information about loaded models"""
        info = {
            "tone_model_loaded": self.tone_model is not None,
            "risk_model_loaded": self.risk_model is not None,
            "metadata_available": self.metadata is not None
        }
        
        if self.metadata:
            info["training_date"] = self.metadata.get("training_date", "Unknown")
            info["total_training_samples"] = self.metadata.get("total_samples", "Unknown")
            info["tone_categories"] = self.metadata.get("tone_categories", [])
        
        return info

# Global AI instance (will be imported by backend)
trap_ai = TrapSiteAI()

def test_ai_inference():
    """Test function for AI inference"""
    test_messages = [
        "hey beautiful how old are you?",
        "your computer has been infected call us now",
        "what's your favorite movie?",
        "send me your bank details for verification",
        "if you don't comply I'll find you"
    ]
    
    print("🧪 Testing AI Inference")
    print("=" * 40)
    
    for message in test_messages:
        result = trap_ai.analyze_message(message)
        print(f"\nMessage: '{message}'")
        print(f"Risk Score: {result['risk_score']}/100")
        print(f"Tone: {result['tone_label']} (confidence: {result['confidence']})")
        print(f"Category: {result['threat_category']}")

if __name__ == "__main__":
    # Print model info
    info = trap_ai.get_model_info()
    print("🤖 AI Model Status:")
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    # Run test if models are loaded
    if info["tone_model_loaded"] and info["risk_model_loaded"]:
        test_ai_inference()
    else:
        print("\n⚠️ Models not fully loaded. Run train_model.py first.")
