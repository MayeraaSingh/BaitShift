"""
Data loader and validation utilities for BaitShift AI model training
"""

import json
import pandas as pd
from collections import Counter

def load_training_data(file_path='dataset.json'):
    """Load training data from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        df = pd.DataFrame(data)
        
        print(f"✅ Loaded {len(df)} training samples")
        print(f"📊 Tone distribution: {dict(df['tone_label'].value_counts())}")
        print(f"📈 Risk score range: {df['risk_score'].min()}-{df['risk_score'].max()}")
        
        return df
        
    except FileNotFoundError:
        print(f"❌ Training data file not found: {file_path}")
        print("Please create training_data.json with your dataset")
        return None
    except Exception as e:
        print(f"❌ Error loading training data: {e}")
        return None

def validate_data_quality(df):
    """Check if dataset is ready for training"""
    if df is None:
        return False
    
    print("\n🔍 Data Quality Check:")
    print("=" * 30)
    
    # Check minimum samples
    total_samples = len(df)
    min_samples_needed = 100
    
    if total_samples < min_samples_needed:
        print(f"❌ Need at least {min_samples_needed} samples, have {total_samples}")
        return False
    else:
        print(f"✅ Sample count: {total_samples}")
    
    # Check tone balance
    tone_counts = df['tone_label'].value_counts()
    min_per_tone = 10
    
    print(f"\n📊 Tone distribution:")
    for tone, count in tone_counts.items():
        status = "✅" if count >= min_per_tone else "❌"
        print(f"   {status} {tone}: {count} samples")
    
    # Check required fields
    required_fields = ['message', 'risk_score', 'tone_label']
    optional_fields = ['threat_category']
    missing_fields = [field for field in required_fields if field not in df.columns]
    
    if missing_fields:
        print(f"❌ Missing required fields: {missing_fields}")
        return False
    
    # Check optional fields
    missing_optional = [field for field in optional_fields if field not in df.columns]
    if missing_optional:
        print(f"⚠️ Missing optional fields: {missing_optional} (will be auto-generated)")
    
    # Check for empty messages
    empty_messages = df['message'].isna().sum() + (df['message'] == '').sum()
    if empty_messages > 0:
        print(f"❌ Found {empty_messages} empty messages")
        return False
    
    # Check risk score range
    invalid_scores = df[(df['risk_score'] < 0) | (df['risk_score'] > 100)]
    if len(invalid_scores) > 0:
        print(f"❌ Found {len(invalid_scores)} invalid risk scores (should be 0-100)")
        return False
    
    print(f"✅ Data quality check passed!")
    return True

def get_data_summary(df):
    """Print comprehensive data summary"""
    if df is None:
        return
    
    print("\n📋 Dataset Summary:")
    print("=" * 40)
    
    print(f"Total samples: {len(df)}")
    print(f"Unique messages: {df['message'].nunique()}")
    print(f"Average message length: {df['message'].str.len().mean():.1f} characters")
    
    print(f"\n📊 Risk Score Distribution:")
    print(f"   Low (0-33): {len(df[df['risk_score'] <= 33])} samples")
    print(f"   Medium (34-66): {len(df[(df['risk_score'] > 33) & (df['risk_score'] <= 66)])} samples")
    print(f"   High (67-100): {len(df[df['risk_score'] > 66])} samples")
    
    print(f"\n🎯 Threat Categories:")
    if 'threat_category' in df.columns:
        for category, count in df['threat_category'].value_counts().items():
            print(f"   {category}: {count}")
    else:
        print("   ⚠️ Threat categories will be auto-generated during training")

if __name__ == "__main__":
    # Test data loading
    df = load_training_data()
    if df is not None:
        validate_data_quality(df)
        get_data_summary(df)
