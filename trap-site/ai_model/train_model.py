"""
Main training script for BaitShift AI models
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
import torch
from transformers import (
    DistilBertTokenizer, 
    DistilBertForSequenceClassification,
    TrainingArguments, 
    Trainer,
    DistilBertConfig
)
from torch.utils.data import Dataset
import pickle
import os
import json
import re
from datetime import datetime
from data_loader import load_training_data, validate_data_quality, get_data_summary

def normalize_text(text):
    """Normalize text for consistent processing"""
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

def auto_generate_threat_category(row):
    """Auto-generate threat category based on tone and risk score"""
    message = row['message'].lower()
    tone = row['tone_label']
    risk = row['risk_score']
    
    # Keyword-based categorization
    if any(word in message for word in ['beautiful', 'cute', 'age', 'old are you', 'meet', 'pic']):
        return "Romance_Scam"
    elif any(word in message for word in ['computer', 'virus', 'infected', 'microsoft', 'download']):
        return "Tech_Support_Scam"
    elif any(word in message for word in ['account', 'verify', 'click', 'link', 'password']):
        return "Phishing"
    elif any(word in message for word in ['money', 'bank', 'dollars', 'invest', 'rich']):
        return "Financial_Scam"
    elif tone == "Threatening" or any(word in message for word in ['find you', 'hurt', 'regret']):
        return "Threat"
    elif any(word in message for word in ['school', 'live', 'home', 'parents']):
        return "Information_Gathering"
    
    # Fallback based on risk score and tone
    if risk >= 80:
        return "High_Risk_Unknown"
    elif risk >= 50:
        return "Medium_Risk_Unknown"
    else:
        return "Low_Risk_Social"

# Create directories
os.makedirs('models', exist_ok=True)
os.makedirs('logs', exist_ok=True)

class AttackerMessageDataset(Dataset):
    """Custom dataset for attacker message classification"""
    
    def __init__(self, messages, tone_labels, tokenizer, max_length=128):
        self.messages = messages
        self.tone_labels = tone_labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, idx):
        message = str(self.messages[idx])
        encoding = self.tokenizer(
            message,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.tone_labels[idx], dtype=torch.long)
        }

def train_tone_classification_model(df):
    """Train DistilBERT model for tone classification"""
    print("\n🤖 Training Tone Classification Model...")
    print("=" * 50)
    
    # Prepare data
    messages = df['message'].tolist()
    
    # Encode tone labels
    tone_encoder = LabelEncoder()
    tone_labels = tone_encoder.fit_transform(df['tone_label']).tolist()
    
    # Save label encoder
    with open('models/tone_encoder.pkl', 'wb') as f:
        pickle.dump(tone_encoder, f)
    
    print(f"📋 Tone categories: {list(tone_encoder.classes_)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        messages, tone_labels, test_size=0.2, random_state=42, stratify=tone_labels
    )
    
    print(f"📊 Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Initialize tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    # Create datasets
    train_dataset = AttackerMessageDataset(X_train, y_train, tokenizer)
    test_dataset = AttackerMessageDataset(X_test, y_test, tokenizer)
    
    # Configure model
    tone_config = DistilBertConfig.from_pretrained(
        'distilbert-base-uncased',
        num_labels=len(tone_encoder.classes_)
    )
    
    tone_model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        config=tone_config
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir='./models/tone_model',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=50,
    )
    
    def compute_metrics_tone(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return {
            'accuracy': accuracy_score(labels, predictions),
            'f1': classification_report(labels, predictions, output_dict=True)['weighted avg']['f1-score']
        }
    
    trainer = Trainer(
        model=tone_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics_tone,
    )
    
    # Train the model
    print("🚀 Starting tone classification training...")
    trainer.train()
    
    # Evaluate final model
    eval_results = trainer.evaluate()
    print(f"✅ Tone Model - Final Accuracy: {eval_results['eval_accuracy']:.4f}")
    print(f"✅ Tone Model - F1 Score: {eval_results['eval_f1']:.4f}")
    
    # Save tone model and tokenizer
    tone_model.save_pretrained('models/tone_model')
    tokenizer.save_pretrained('models/tone_tokenizer')
    
    print("💾 Tone classification model saved!")
    return tone_encoder.classes_, eval_results['eval_accuracy'], eval_results['eval_f1']

def train_risk_score_model(df):
    """Train Random Forest model for risk score prediction"""
    print("\n📊 Training Risk Score Model...")
    print("=" * 50)
    
    # Prepare data
    messages = df['message'].tolist()
    risk_scores = (df['risk_score'] / 100.0).tolist()  # Normalize to 0-1
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        messages, risk_scores, test_size=0.2, random_state=42
    )
    
    print(f"📊 Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Use TF-IDF vectorization
    vectorizer = TfidfVectorizer(
        max_features=2000, 
        stop_words=None,  # Don't remove stop words - they're important for threat detection
        ngram_range=(1, 3),  # Include trigrams for better context
        min_df=1,  # Include rare words
        lowercase=True
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Train Random Forest regressor
    risk_model = RandomForestRegressor(
        n_estimators=50,  # Reduce trees to prevent overfitting
        random_state=42,
        max_depth=3,      # Reduce depth for small dataset
        min_samples_split=2,  # Allow smaller splits
        min_samples_leaf=2    # Minimum samples per leaf
    )
    
    print("🚀 Training risk score model...")
    risk_model.fit(X_train_tfidf, y_train)
    
    # Evaluate model
    y_pred = risk_model.predict(X_test_tfidf)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Convert back to 0-100 scale for interpretation
    y_test_scaled = [score * 100 for score in y_test]
    y_pred_scaled = [score * 100 for score in y_pred]
    mae = np.mean([abs(actual - predicted) for actual, predicted in zip(y_test_scaled, y_pred_scaled)])
    
    print(f"✅ Risk Model Performance:")
    print(f"   Mean Squared Error: {mse:.4f}")
    print(f"   R² Score: {r2:.4f}")
    print(f"   Mean Absolute Error (0-100 scale): {mae:.2f} points")
    
    # Show some example predictions
    print(f"\n🎯 Example Predictions:")
    for i in range(min(5, len(X_test))):
        actual = y_test_scaled[i]
        predicted = y_pred_scaled[i]
        message = X_test[i][:50] + "..." if len(X_test[i]) > 50 else X_test[i]
        print(f"   Message: '{message}'")
        print(f"   Actual: {actual:.1f}, Predicted: {predicted:.1f}")
        print()
    
    # Save risk model and vectorizer
    with open('models/risk_model.pkl', 'wb') as f:
        pickle.dump(risk_model, f)
    
    with open('models/risk_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print("💾 Risk score model saved!")
    return mse, r2, mae

def save_training_metadata(tone_classes, tone_accuracy, tone_f1, risk_metrics, total_samples):
    """Save training metadata for reference"""
    metadata = {
        "training_date": datetime.now().isoformat(),
        "total_samples": total_samples,
        "tone_categories": list(tone_classes),
        "tone_model_metrics": {
            "accuracy": float(tone_accuracy),
            "f1_score": float(tone_f1)
        },
        "risk_model_metrics": {
            "mse": float(risk_metrics[0]),
            "r2_score": float(risk_metrics[1]),
            "mae_0_100_scale": float(risk_metrics[2])
        },
        "model_info": {
            "tone_model": "DistilBERT-base-uncased",
            "risk_model": "RandomForestRegressor",
            "tokenizer": "DistilBERT tokenizer",
            "max_sequence_length": 128
        }
    }
    
    with open('models/training_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("📋 Training metadata saved!")

def main():
    """Main training pipeline"""
    print("🎯 BaitShift AI Model Training Pipeline")
    print("=" * 60)
    
    # Load and validate data
    df = load_training_data()
    if df is None:
        print("❌ Cannot proceed without training data")
        return
    
    # Normalize text for consistent processing
    print("🔧 Normalizing text data...")
    df['message'] = df['message'].apply(normalize_text)
    print("✅ Text normalization complete")
    
    # Auto-generate threat categories if missing
    if 'threat_category' not in df.columns:
        print("⚠️ Generating threat categories based on tone and risk score...")
        df['threat_category'] = df.apply(auto_generate_threat_category, axis=1)
        print("✅ Threat categories generated")

    if not validate_data_quality(df):
        print("❌ Data quality check failed")
        return

    get_data_summary(df)
    
    # Train tone classification model
    tone_classes, tone_accuracy, tone_f1 = train_tone_classification_model(df)
    
    # Train risk score model
    risk_metrics = train_risk_score_model(df)
    
    # Save metadata
    save_training_metadata(tone_classes, tone_accuracy, tone_f1, risk_metrics, len(df))
    
    print("\n🎉 Training Complete!")
    print("=" * 60)
    print("✅ Models trained and saved:")
    print("   📁 models/tone_model/ (DistilBERT)")
    print("   📁 models/risk_model.pkl (Random Forest)")
    print("   📁 models/tone_encoder.pkl")
    print("   📁 models/risk_vectorizer.pkl")
    print("   📁 models/tone_tokenizer/")
    print("   📋 models/training_metadata.json")
    
    print(f"\n🎯 Model Performance Summary:")
    print(f"   Tone Classification Accuracy: {tone_accuracy:.3f}")
    print(f"   Tone Classification F1: {tone_f1:.3f}")
    print(f"   Risk Score MAE: {risk_metrics[2]:.1f} points (0-100 scale)")
    print(f"   Risk Score R²: {risk_metrics[1]:.3f}")
    
    print("\n🚀 Ready for integration with trap site backend!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print("Check your environment and dependencies.")
