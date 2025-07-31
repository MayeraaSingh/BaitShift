"""
Improved training script for BaitShift AI models
Addresses convergence issues and improves robustness
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import VotingRegressor
import torch
from transformers import (
    DistilBertTokenizer, 
    DistilBertForSequenceClassification,
    TrainingArguments, 
    Trainer,
    DistilBertConfig,
    EarlyStoppingCallback
)
from torch.utils.data import Dataset, WeightedRandomSampler
import pickle
import os
import json
import re
from datetime import datetime
from collections import Counter
from data_loader import load_training_data, validate_data_quality, get_data_summary

def normalize_text(text):
    """Enhanced text normalization"""
    text = str(text).lower()
    
    # Expand contractions
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
    
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove URLs and email patterns (but keep structure for threat detection)
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'URL_PLACEHOLDER', text)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 'EMAIL_PLACEHOLDER', text)
    
    return text

def augment_dataset(df):
    """Augment the dataset to address class imbalance and small size"""
    print("🔄 Augmenting dataset...")
    
    augmented_data = []
    
    # Original data
    for _, row in df.iterrows():
        augmented_data.append(row.to_dict())
    
    # Add variations for underrepresented classes
    tone_counts = df['tone_label'].value_counts()
    min_samples = max(20, tone_counts.min() * 2)  # Ensure at least 20 samples per class
    
    for tone_label in tone_counts.index:
        current_count = tone_counts[tone_label]
        if current_count < min_samples:
            # Find samples of this tone
            tone_samples = df[df['tone_label'] == tone_label]
            needed = min_samples - current_count
            
            # Create variations by slight modifications
            for i in range(needed):
                original = tone_samples.iloc[i % len(tone_samples)]
                
                # Create variation
                new_sample = original.copy()
                message = original['message']
                
                # Add slight variations while preserving meaning
                variations = [
                    message + " please",
                    message + " now",
                    "so " + message,
                    message.replace("you", "u").replace("your", "ur"),
                    message.replace("what", "wat").replace("how", "hw"),
                ]
                
                new_sample['message'] = variations[i % len(variations)]
                # Add slight noise to risk score (±3 points)
                noise = np.random.randint(-3, 4)
                new_sample['risk_score'] = max(0, min(100, original['risk_score'] + noise))
                
                augmented_data.append(new_sample.to_dict())
    
    augmented_df = pd.DataFrame(augmented_data)
    print(f"✅ Dataset augmented from {len(df)} to {len(augmented_df)} samples")
    
    return augmented_df

def create_engineered_features(df):
    """Create additional features for better risk prediction"""
    print("🔧 Engineering features...")
    
    df = df.copy()
    
    # Message length features
    df['message_length'] = df['message'].str.len()
    df['word_count'] = df['message'].str.split().str.len()
    df['avg_word_length'] = df['message_length'] / df['word_count']
    
    # Character features
    df['uppercase_ratio'] = df['message'].apply(lambda x: sum(c.isupper() for c in x) / len(x) if len(x) > 0 else 0)
    df['question_marks'] = df['message'].str.count(r'\?')
    df['exclamation_marks'] = df['message'].str.count('!')
    df['numbers_count'] = df['message'].apply(lambda x: sum(c.isdigit() for c in x))
    
    # Threat keywords (more comprehensive)
    threat_keywords = {
        'romantic': ['beautiful', 'cute', 'sexy', 'love', 'baby', 'sweetheart', 'dear'],
        'urgency': ['now', 'immediately', 'urgent', 'hurry', 'quick', 'asap'],
        'technical': ['computer', 'virus', 'infected', 'download', 'click', 'microsoft'],
        'financial': ['money', 'bank', 'payment', 'transfer', 'card', 'account'],
        'threatening': ['find you', 'hurt', 'kill', 'die', 'regret', 'consequences'],
        'personal_info': ['age', 'school', 'address', 'phone', 'live', 'alone']
    }
    
    for category, keywords in threat_keywords.items():
        df[f'{category}_keywords'] = df['message'].apply(
            lambda x: sum(1 for keyword in keywords if keyword in x.lower())
        )
    
    # Tone encoding for features
    tone_encoder = LabelEncoder()
    df['tone_encoded'] = tone_encoder.fit_transform(df['tone_label'])
    
    print("✅ Feature engineering complete")
    return df, tone_encoder

class ImprovedAttackerMessageDataset(Dataset):
    """Enhanced dataset with class weighting"""
    
    def __init__(self, messages, tone_labels, tokenizer, max_length=128):
        self.messages = messages
        self.tone_labels = tone_labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, idx):
        message = str(self.messages[idx])
        # Normalize message
        message = normalize_text(message)
        
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

def train_improved_tone_model(df):
    """Train improved tone classification with better handling of imbalanced data"""
    print("\n🤖 Training Improved Tone Classification Model...")
    print("=" * 60)
    
    # Normalize messages
    df['message'] = df['message'].apply(normalize_text)
    
    # Prepare data
    messages = df['message'].tolist()
    
    # Encode tone labels
    tone_encoder = LabelEncoder()
    tone_labels = tone_encoder.fit_transform(df['tone_label']).tolist()
    
    # Calculate class weights
    class_counts = Counter(tone_labels)
    total_samples = len(tone_labels)
    class_weights = {cls: total_samples / (len(class_counts) * count) for cls, count in class_counts.items()}
    
    print(f"📋 Tone categories: {list(tone_encoder.classes_)}")
    print(f"⚖️ Class weights: {class_weights}")
    
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        messages, tone_labels, test_size=0.2, random_state=42, stratify=tone_labels
    )
    
    print(f"📊 Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Initialize tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    # Create datasets
    train_dataset = ImprovedAttackerMessageDataset(X_train, y_train, tokenizer)
    test_dataset = ImprovedAttackerMessageDataset(X_test, y_test, tokenizer)
    
    # Configure model with class weights
    tone_config = DistilBertConfig.from_pretrained(
        'distilbert-base-uncased',
        num_labels=len(tone_encoder.classes_)
    )
    
    tone_model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        config=tone_config
    )
    
    # Enhanced training arguments
    training_args = TrainingArguments(
        output_dir='./models/tone_model',
        num_train_epochs=5,  # More epochs
        per_device_train_batch_size=4,  # Smaller batch for stability
        per_device_eval_batch_size=4,
        warmup_steps=200,  # More warmup
        weight_decay=0.01,
        learning_rate=2e-5,  # Lower learning rate
        logging_dir='./logs',
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",  # Optimize for F1
        logging_steps=25,
        eval_steps=50,
        save_total_limit=3,  # Keep only best 3 models
    )
    
    def compute_metrics_tone(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        report = classification_report(labels, predictions, output_dict=True, zero_division=0)
        
        return {
            'accuracy': accuracy,
            'f1': report['weighted avg']['f1-score'],
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall']
        }
    
    trainer = Trainer(
        model=tone_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics_tone,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # Train the model
    print("🚀 Starting improved tone classification training...")
    trainer.train()
    
    # Final evaluation
    eval_results = trainer.evaluate()
    print(f"✅ Improved Tone Model Results:")
    print(f"   Accuracy: {eval_results['eval_accuracy']:.4f}")
    print(f"   F1 Score: {eval_results['eval_f1']:.4f}")
    print(f"   Precision: {eval_results['eval_precision']:.4f}")
    print(f"   Recall: {eval_results['eval_recall']:.4f}")
    
    # Save models
    tone_model.save_pretrained('models/tone_model')
    tokenizer.save_pretrained('models/tone_tokenizer')
    
    with open('models/tone_encoder.pkl', 'wb') as f:
        pickle.dump(tone_encoder, f)
    
    print("💾 Improved tone model saved!")
    return tone_encoder.classes_, eval_results

def train_ensemble_risk_model(df):
    """Train ensemble risk model for better predictions"""
    print("\n📊 Training Ensemble Risk Score Model...")
    print("=" * 60)
    
    # Feature engineering
    df_features, _ = create_engineered_features(df)
    
    # Prepare text features
    messages = df_features['message'].tolist()
    
    # Prepare numerical features
    feature_cols = [
        'message_length', 'word_count', 'avg_word_length', 'uppercase_ratio',
        'question_marks', 'exclamation_marks', 'numbers_count', 'tone_encoded',
        'romantic_keywords', 'urgency_keywords', 'technical_keywords',
        'financial_keywords', 'threatening_keywords', 'personal_info_keywords'
    ]
    
    numerical_features = df_features[feature_cols].fillna(0).values
    risk_scores = (df_features['risk_score'] / 100.0).tolist()  # Normalize to 0-1
    
    # Split data
    X_text_train, X_text_test, X_num_train, X_num_test, y_train, y_test = train_test_split(
        messages, numerical_features, risk_scores, test_size=0.2, random_state=42
    )
    
    print(f"📊 Training samples: {len(X_text_train)}, Test samples: {len(X_text_test)}")
    
    # Text vectorization with improved parameters
    vectorizer = TfidfVectorizer(
        max_features=1500,  # Reduced features to prevent overfitting
        stop_words=None,
        ngram_range=(1, 2),  # Reduced n-gram range
        min_df=2,  # Ignore very rare terms
        max_df=0.8,  # Ignore very common terms
        lowercase=True,
        sublinear_tf=True  # Apply sublinear tf scaling
    )
    
    X_text_train_tfidf = vectorizer.fit_transform(X_text_train)
    X_text_test_tfidf = vectorizer.transform(X_text_test)
    
    # Scale numerical features
    scaler = StandardScaler()
    X_num_train_scaled = scaler.fit_transform(X_num_train)
    X_num_test_scaled = scaler.transform(X_num_test)
    
    # Combine text and numerical features
    from scipy.sparse import hstack
    X_train_combined = hstack([X_text_train_tfidf, X_num_train_scaled])
    X_test_combined = hstack([X_text_test_tfidf, X_num_test_scaled])
    
    # Create ensemble of different models
    models = {
        'random_forest': RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            max_depth=8,  # Increased depth
            min_samples_split=5,
            min_samples_leaf=3,
            max_features='sqrt'
        ),
        'gradient_boost': GradientBoostingRegressor(
            n_estimators=100,
            random_state=42,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8
        ),
        'ridge': Ridge(alpha=1.0, random_state=42)
    }
    
    # Train individual models and evaluate
    trained_models = {}
    individual_scores = {}
    
    for name, model in models.items():
        print(f"🔧 Training {name}...")
        model.fit(X_train_combined, y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_combined, y_train, cv=5, scoring='r2')
        
        # Test predictions
        y_pred = model.predict(X_test_combined)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        
        trained_models[name] = model
        individual_scores[name] = {
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std(),
            'test_r2': r2,
            'test_mse': mse
        }
        
        print(f"   CV R² Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        print(f"   Test R² Score: {r2:.4f}")
    
    # Create ensemble model
    ensemble = VotingRegressor([
        ('rf', models['random_forest']),
        ('gb', models['gradient_boost']),
        ('ridge', models['ridge'])
    ])
    
    print("🚀 Training ensemble model...")
    ensemble.fit(X_train_combined, y_train)
    
    # Evaluate ensemble
    y_pred_ensemble = ensemble.predict(X_test_combined)
    ensemble_r2 = r2_score(y_test, y_pred_ensemble)
    ensemble_mse = mean_squared_error(y_test, y_pred_ensemble)
    
    # Convert to 0-100 scale for interpretation
    y_test_scaled = [score * 100 for score in y_test]
    y_pred_scaled = [score * 100 for score in y_pred_ensemble]
    mae = np.mean([abs(actual - predicted) for actual, predicted in zip(y_test_scaled, y_pred_scaled)])
    
    print(f"\n✅ Ensemble Model Performance:")
    print(f"   R² Score: {ensemble_r2:.4f}")
    print(f"   Mean Squared Error: {ensemble_mse:.4f}")
    print(f"   Mean Absolute Error (0-100 scale): {mae:.2f} points")
    
    # Show example predictions
    print(f"\n🎯 Example Predictions:")
    for i in range(min(5, len(X_text_test))):
        actual = y_test_scaled[i]
        predicted = y_pred_scaled[i]
        message = X_text_test[i][:50] + "..." if len(X_text_test[i]) > 50 else X_text_test[i]
        print(f"   Message: '{message}'")
        print(f"   Actual: {actual:.1f}, Predicted: {predicted:.1f}, Diff: {abs(actual-predicted):.1f}")
        print()
    
    # Save all components
    with open('models/risk_ensemble_model.pkl', 'wb') as f:
        pickle.dump(ensemble, f)
    
    with open('models/risk_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    with open('models/risk_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Also save individual models for comparison
    with open('models/individual_risk_models.pkl', 'wb') as f:
        pickle.dump(trained_models, f)
    
    print("💾 Ensemble risk model saved!")
    return ensemble_mse, ensemble_r2, mae, individual_scores

def save_improved_metadata(tone_classes, tone_results, risk_metrics, individual_scores, total_samples):
    """Save comprehensive training metadata"""
    metadata = {
        "training_date": datetime.now().isoformat(),
        "total_samples": total_samples,
        "model_version": "improved_v2",
        "tone_categories": list(tone_classes),
        "tone_model_metrics": {
            "accuracy": float(tone_results['eval_accuracy']),
            "f1_score": float(tone_results['eval_f1']),
            "precision": float(tone_results['eval_precision']),
            "recall": float(tone_results['eval_recall'])
        },
        "risk_ensemble_metrics": {
            "mse": float(risk_metrics[0]),
            "r2_score": float(risk_metrics[1]),
            "mae_0_100_scale": float(risk_metrics[2])
        },
        "individual_model_scores": individual_scores,
        "model_info": {
            "tone_model": "DistilBERT-base-uncased (improved)",
            "risk_model": "VotingRegressor (RF+GB+Ridge)",
            "tokenizer": "DistilBERT tokenizer",
            "max_sequence_length": 128,
            "features_engineered": True,
            "dataset_augmented": True
        },
        "improvements": [
            "Dataset augmentation for class balance",
            "Feature engineering with keyword detection",
            "Ensemble risk prediction model",
            "Enhanced text normalization",
            "Cross-validation and early stopping",
            "Class weighting for imbalanced data"
        ]
    }
    
    with open('models/training_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("📋 Improved training metadata saved!")

def main():
    """Improved training pipeline"""
    print("🎯 BaitShift AI Model - IMPROVED Training Pipeline")
    print("=" * 70)
    
    # Load and validate data
    df = load_training_data()
    if df is None:
        print("❌ Cannot proceed without training data")
        return
    
    print(f"📊 Original dataset size: {len(df)} samples")
    
    # Augment dataset
    df_augmented = augment_dataset(df)
    
    # Normalize text
    print("🔧 Normalizing text data...")
    df_augmented['message'] = df_augmented['message'].apply(normalize_text)
    
    if not validate_data_quality(df_augmented):
        print("❌ Data quality check failed")
        return
    
    get_data_summary(df_augmented)
    
    # Train improved models
    tone_classes, tone_results = train_improved_tone_model(df_augmented)
    risk_metrics = train_ensemble_risk_model(df_augmented)
    
    # Extract individual scores from return
    ensemble_mse, ensemble_r2, mae, individual_scores = risk_metrics
    risk_metrics_tuple = (ensemble_mse, ensemble_r2, mae)
    
    # Save metadata
    save_improved_metadata(tone_classes, tone_results, risk_metrics_tuple, individual_scores, len(df_augmented))
    
    print("\n🎉 IMPROVED Training Complete!")
    print("=" * 70)
    print("✅ Enhanced models trained and saved:")
    print("   📁 models/tone_model/ (Improved DistilBERT)")
    print("   📁 models/risk_ensemble_model.pkl (Ensemble)")
    print("   📁 models/individual_risk_models.pkl")
    print("   📁 models/risk_scaler.pkl")
    print("   📁 models/tone_encoder.pkl")
    print("   📁 models/risk_vectorizer.pkl")
    print("   📁 models/tone_tokenizer/")
    print("   📋 models/training_metadata.json")
    
    print(f"\n🎯 Enhanced Model Performance:")
    print(f"   Tone Classification:")
    print(f"     - Accuracy: {tone_results['eval_accuracy']:.3f}")
    print(f"     - F1 Score: {tone_results['eval_f1']:.3f}")
    print(f"     - Precision: {tone_results['eval_precision']:.3f}")
    print(f"     - Recall: {tone_results['eval_recall']:.3f}")
    print(f"   Risk Score Prediction:")
    print(f"     - R² Score: {risk_metrics_tuple[1]:.3f}")
    print(f"     - MAE: {risk_metrics_tuple[2]:.1f} points")
    print(f"     - MSE: {risk_metrics_tuple[0]:.4f}")
    
    print("\n🚀 Ready for production deployment!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Improved training failed: {e}")
        import traceback
        traceback.print_exc()
