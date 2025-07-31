"""
Improved AI inference module for BaitShift trap site
Provides robust real-time threat analysis with ensemble models
"""

import torch
import pickle
import numpy as np
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import os
import json
import re
from scipy.sparse import hstack
from sklearn.preprocessing import StandardScaler

def normalize_text(text):
    """Enhanced text normalization - matches improved training"""
    text = str(text).lower()
    
    # Expanded contractions
    contractions = {
        "you're": "you are", "we're": "we are", "they're": "they are",
        "can't": "cannot", "won't": "will not", "don't": "do not",
        "i'm": "i am", "i'll": "i will", "i've": "i have",
        "what's": "what is", "that's": "that is", "there's": "there is",
        "it's": "it is", "here's": "here is", "where's": "where is",
        "how's": "how is", "let's": "let us", "who's": "who is"
    }
    
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)
    
    # Remove URLs and emails (replace with placeholders)
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'URL_PLACEHOLDER', text)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 'EMAIL_PLACEHOLDER', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_features(message):
    """Extract engineered features from message"""
    # Message length features
    message_length = len(message)
    words = message.split()
    word_count = len(words)
    avg_word_length = message_length / word_count if word_count > 0 else 0
    
    # Character features
    uppercase_ratio = sum(c.isupper() for c in message) / len(message) if len(message) > 0 else 0
    question_marks = message.count('?')
    exclamation_marks = message.count('!')
    numbers_count = sum(c.isdigit() for c in message)
    
    # Threat keywords
    message_lower = message.lower()
    
    threat_keywords = {
        'romantic': ['beautiful', 'cute', 'sexy', 'love', 'baby', 'sweetheart', 'dear'],
        'urgency': ['now', 'immediately', 'urgent', 'hurry', 'quick', 'asap'],
        'technical': ['computer', 'virus', 'infected', 'download', 'click', 'microsoft'],
        'financial': ['money', 'bank', 'payment', 'transfer', 'card', 'account'],
        'threatening': ['find you', 'hurt', 'kill', 'die', 'regret', 'consequences'],
        'personal_info': ['age', 'school', 'address', 'phone', 'live', 'alone']
    }
    
    keyword_counts = {}
    for category, keywords in threat_keywords.items():
        keyword_counts[f'{category}_keywords'] = sum(1 for keyword in keywords if keyword in message_lower)
    
    # Combine all features
    features = [
        message_length, word_count, avg_word_length, uppercase_ratio,
        question_marks, exclamation_marks, numbers_count,
        0,  # tone_encoded - will be filled after tone prediction
        keyword_counts['romantic_keywords'],
        keyword_counts['urgency_keywords'],
        keyword_counts['technical_keywords'],
        keyword_counts['financial_keywords'],
        keyword_counts['threatening_keywords'],
        keyword_counts['personal_info_keywords']
    ]
    
    return np.array(features).reshape(1, -1)

class ImprovedTrapSiteAI:
    def __init__(self, models_dir=None):
        # Set models directory relative to this file's location
        if models_dir is None:
            # Get the directory where this file is located
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.models_dir = os.path.join(current_dir, 'models')
        else:
            self.models_dir = models_dir
        
        self.tone_model = None
        self.tone_tokenizer = None
        self.tone_encoder = None
        self.risk_ensemble_model = None
        self.risk_vectorizer = None
        self.risk_scaler = None
        self.metadata = None
        self.fallback_to_simple = False
        self.load_models()
    
    def load_models(self):
        """Load improved models with fallback to simple models"""
        try:
            # Load metadata
            metadata_path = os.path.join(self.models_dir, 'training_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                model_version = self.metadata.get('model_version', 'v1')
                print(f"✅ Loaded training metadata - Version: {model_version}")
            
            # Try to load improved models first
            self._load_improved_models()
            
            # Fallback to simple models if improved ones don't exist
            if not (self.tone_model and self.risk_ensemble_model):
                print("⚠️ Improved models not found, falling back to simple models")
                self.fallback_to_simple = True
                self._load_simple_models()
            
        except Exception as e:
            print(f"⚠️ Error loading models: {e}")
            print("🔄 Models will return default values")
    
    def _load_improved_models(self):
        """Load improved ensemble models"""
        try:
            # Load tone model (same as before but improved)
            tone_model_path = os.path.join(self.models_dir, 'tone_model')
            tone_tokenizer_path = os.path.join(self.models_dir, 'tone_tokenizer')
            tone_encoder_path = os.path.join(self.models_dir, 'tone_encoder.pkl')
            
            if all(os.path.exists(p) for p in [tone_model_path, tone_encoder_path]):
                self.tone_model = DistilBertForSequenceClassification.from_pretrained(tone_model_path)
                self.tone_tokenizer = DistilBertTokenizer.from_pretrained(tone_tokenizer_path)
                
                with open(tone_encoder_path, 'rb') as f:
                    self.tone_encoder = pickle.load(f)
                
                self.tone_model.eval()
                print("✅ Improved tone classification model loaded")
            
            # Load ensemble risk model
            ensemble_path = os.path.join(self.models_dir, 'risk_ensemble_model.pkl')
            vectorizer_path = os.path.join(self.models_dir, 'risk_vectorizer.pkl')
            scaler_path = os.path.join(self.models_dir, 'risk_scaler.pkl')
            
            if all(os.path.exists(p) for p in [ensemble_path, vectorizer_path, scaler_path]):
                with open(ensemble_path, 'rb') as f:
                    self.risk_ensemble_model = pickle.load(f)
                
                with open(vectorizer_path, 'rb') as f:
                    self.risk_vectorizer = pickle.load(f)
                
                with open(scaler_path, 'rb') as f:
                    self.risk_scaler = pickle.load(f)
                
                print("✅ Improved ensemble risk model loaded")
            
        except Exception as e:
            print(f"⚠️ Could not load improved models: {e}")
    
    def _load_simple_models(self):
        """Fallback to simple models"""
        try:
            # Load simple risk model
            simple_risk_path = os.path.join(self.models_dir, 'risk_model.pkl')
            if os.path.exists(simple_risk_path):
                with open(simple_risk_path, 'rb') as f:
                    self.risk_ensemble_model = pickle.load(f)
                print("✅ Simple risk model loaded as fallback")
            
        except Exception as e:
            print(f"⚠️ Could not load simple models: {e}")
    
    def predict_tone(self, message):
        """Predict tone with improved confidence"""
        if not self.tone_model or not self.tone_tokenizer or not self.tone_encoder:
            return {"tone_label": "Unknown", "confidence": 0.0}
        
        try:
            normalized_message = normalize_text(message)
            
            inputs = self.tone_tokenizer(
                normalized_message,
                truncation=True,
                padding='max_length',
                max_length=128,
                return_tensors='pt'
            )
            
            with torch.no_grad():
                outputs = self.tone_model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(predictions, dim=-1).item()
                confidence = predictions[0][predicted_class].item()
            
            tone_label = self.tone_encoder.inverse_transform([predicted_class])[0]
            
            return {
                "tone_label": tone_label,
                "confidence": round(confidence, 3)
            }
            
        except Exception as e:
            print(f"❌ Error predicting tone: {e}")
            return {"tone_label": "Unknown", "confidence": 0.0}
    
    def predict_risk_score(self, message, tone_encoded=0):
        """Predict risk score using ensemble model"""
        if not self.risk_ensemble_model or not self.risk_vectorizer:
            return self._fallback_risk_prediction(message)
        
        try:
            normalized_message = normalize_text(message)
            
            # Get text features
            message_tfidf = self.risk_vectorizer.transform([normalized_message])
            
            if not self.fallback_to_simple:
                # Use improved model with engineered features
                numerical_features = extract_features(message)
                numerical_features[0][7] = tone_encoded  # Add tone encoding
                
                if self.risk_scaler:
                    numerical_features_scaled = self.risk_scaler.transform(numerical_features)
                    # Combine text and numerical features
                    combined_features = hstack([message_tfidf, numerical_features_scaled])
                else:
                    combined_features = message_tfidf
                
                risk_normalized = self.risk_ensemble_model.predict(combined_features)[0]
            else:
                # Use simple model
                risk_normalized = self.risk_ensemble_model.predict(message_tfidf)[0]
            
            # Convert to 0-100 scale with bounds checking
            risk_score = max(0, min(100, int(risk_normalized * 100)))
            
            # Apply additional validation to prevent convergence to 50
            if 45 <= risk_score <= 55:
                # Add keyword-based adjustment to break convergence
                risk_score = self._keyword_based_adjustment(message, risk_score)
            
            return risk_score
            
        except Exception as e:
            print(f"❌ Error predicting risk score: {e}")
            return self._fallback_risk_prediction(message)
    
    def _keyword_based_adjustment(self, message, base_score):
        """Adjust risk score based on keyword analysis to prevent convergence"""
        message_lower = message.lower()
        
        high_risk_keywords = [
            'beautiful', 'cute', 'send pic', 'meet', 'virus', 'download',
            'bank', 'money', 'verify', 'click', 'find you', 'hurt'
        ]
        
        low_risk_keywords = [
            'hello', 'hi', 'how are you', 'weather', 'movie', 'book',
            'school', 'homework', 'friend'
        ]
        
        high_risk_count = sum(1 for keyword in high_risk_keywords if keyword in message_lower)
        low_risk_count = sum(1 for keyword in low_risk_keywords if keyword in message_lower)
        
        if high_risk_count > 0:
            adjustment = min(25, high_risk_count * 8)  # Increase by up to 25 points
            return min(100, base_score + adjustment)
        elif low_risk_count > 0:
            adjustment = min(20, low_risk_count * 5)   # Decrease by up to 20 points
            return max(0, base_score - adjustment)
        
        return base_score
    
    def _fallback_risk_prediction(self, message):
        """Fallback risk prediction using rule-based approach"""
        message_lower = message.lower()
        
        # High risk patterns
        if any(pattern in message_lower for pattern in [
            'beautiful', 'cute', 'send pic', 'meet me', 'age', 'virus',
            'download', 'bank', 'password', 'verify', 'click', 'find you'
        ]):
            return np.random.randint(75, 95)  # High risk with variation
        
        # Medium risk patterns
        elif any(pattern in message_lower for pattern in [
            'school', 'live', 'phone', 'address', 'parent'
        ]):
            return np.random.randint(45, 70)  # Medium risk with variation
        
        # Low risk patterns
        else:
            return np.random.randint(5, 35)   # Low risk with variation
    
    def analyze_message(self, message):
        """Complete improved analysis of an attacker message"""
        if not message or not message.strip():
            return {
                "risk_score": 0,
                "tone_label": "Empty",
                "confidence": 1.0,
                "threat_category": "Empty_Message",
                "model_version": self.metadata.get('model_version', 'unknown') if self.metadata else 'unknown'
            }
        
        # Get tone prediction first
        tone_result = self.predict_tone(message)
        tone_label = tone_result["tone_label"]
        confidence = tone_result["confidence"]
        
        # Encode tone for risk prediction
        tone_encoded = 0
        if self.tone_encoder and tone_label != "Unknown":
            try:
                tone_encoded = self.tone_encoder.transform([tone_label])[0]
            except:
                tone_encoded = 0
        
        # Get risk score with tone information
        risk_score = self.predict_risk_score(message, tone_encoded)
        
        result = {
            "risk_score": risk_score,
            "tone_label": tone_label,
            "confidence": confidence,
            "threat_category": self._classify_threat_category(message, tone_label, risk_score),
            "model_version": self.metadata.get('model_version', 'unknown') if self.metadata else 'unknown'
        }
        
        return result
    
    def _classify_threat_category(self, message, tone_label, risk_score):
        """Enhanced rule-based threat categorization"""
        message_lower = message.lower()
        
        # High-confidence categorization
        if any(word in message_lower for word in ['beautiful', 'cute', 'meet', 'private', 'age', 'send pic', 'baby']):
            return "Romance_Scam"
        
        if any(word in message_lower for word in ['computer', 'virus', 'download', 'remote', 'tech', 'microsoft', 'infected']):
            return "Tech_Support_Scam"
        
        if any(word in message_lower for word in ['account', 'verify', 'click', 'link', 'suspended', 'login', 'password']):
            return "Phishing"
        
        if any(word in message_lower for word in ['money', 'bank', 'payment', 'invest', 'crypto', 'rich', 'dollars']):
            return "Financial_Scam"
        
        if any(word in message_lower for word in ['find you', 'know where', 'or else', 'hurt you', 'regret', 'die']):
            return "Threat"
        
        if any(word in message_lower for word in ['school', 'where do you live', 'grade', 'home alone', 'parents']):
            return "Information_Gathering"
        
        # Enhanced fallback based on risk and tone
        if risk_score >= 80:
            if tone_label in ["Threatening", "Urgent"]:
                return "High_Risk_Threat"
            elif tone_label == "Manipulative":
                return "High_Risk_Manipulation"
            else:
                return "High_Risk_Unknown"
        elif risk_score >= 60:
            if tone_label == "Technical":
                return "Medium_Risk_Technical"
            else:
                return "Medium_Risk_Social"
        elif risk_score >= 30:
            return "Low_Risk_Social"
        else:
            return "Benign_Conversation"
    
    def get_model_info(self):
        """Get comprehensive model information"""
        info = {
            "tone_model_loaded": self.tone_model is not None,
            "risk_model_loaded": self.risk_ensemble_model is not None,
            "metadata_available": self.metadata is not None,
            "using_improved_models": not self.fallback_to_simple,
            "model_type": "ensemble" if not self.fallback_to_simple else "simple"
        }
        
        if self.metadata:
            info.update({
                "model_version": self.metadata.get("model_version", "unknown"),
                "training_date": self.metadata.get("training_date", "Unknown"),
                "total_training_samples": self.metadata.get("total_samples", "Unknown"),
                "tone_categories": self.metadata.get("tone_categories", []),
                "improvements": self.metadata.get("improvements", [])
            })
        
        return info

# Global improved AI instance
trap_ai = ImprovedTrapSiteAI()

def test_improved_inference():
    """Test improved AI inference with diverse examples"""
    test_messages = [
        "hey beautiful how old are you?",  # Romance scam
        "your computer has been infected call us now",  # Tech support
        "what's your favorite movie?",  # Benign
        "send me your bank details for verification",  # Financial scam
        "if you don't comply I'll find you",  # Threat
        "what school do you go to",  # Info gathering
        "download this file to fix the problem",  # Tech scam
        "you look so mature for your age",  # Manipulation
        "how was your day today",  # Benign conversation
        "click here to verify your suspended account"  # Phishing
    ]
    
    print("🧪 Testing Improved AI Inference")
    print("=" * 50)
    
    for message in test_messages:
        result = trap_ai.analyze_message(message)
        print(f"\\nMessage: '{message}'")
        print(f"Risk Score: {result['risk_score']}/100")
        print(f"Tone: {result['tone_label']} (confidence: {result['confidence']})")
        print(f"Category: {result['threat_category']}")
        print(f"Model: {result['model_version']}")

if __name__ == "__main__":
    # Print model info
    info = trap_ai.get_model_info()
    print("🤖 Improved AI Model Status:")
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    # Run test
    test_improved_inference()
