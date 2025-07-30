# AI Model Training and Inference

This folder contains the AI model training pipeline for the BaitShift trap site.

## Files

- `dataset.json` - Training dataset (394 samples ready!)
- `data_loader.py` - Data loading and validation utilities
- `train_model.py` - Main training script
- `ai_inference.py` - Model inference for production use
- `test_models.py` - Testing script to verify trained models
- `requirements.txt` - Python dependencies

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Your training data is ready:**
   Your `dataset.json` contains 394 samples with perfect distribution:
   ```json
   [
     {
       "message": "hey beautiful how old r u",
       "risk_score": 80,
       "tone_label": "Manipulative"
     }
   ]
   ```
   (Threat categories will be auto-generated during training)

3. **Train models:**
   ```bash
   python train_model.py
   ```

4. **Test models:**
   ```bash
   python test_models.py
   ```

5. **Use in production:**
   ```python
   from ai_inference import trap_ai
   result = trap_ai.analyze_message("suspicious message")
   ```

## Model Outputs

- `risk_score`: 0-100 danger level
- `tone_label`: Friendly, Urgent, Manipulative, Threatening, Persuasive, Technical
- `threat_category`: Romance_Scam, Tech_Support_Scam, Phishing, Financial_Scam, etc.
- `confidence`: Model confidence (0-1)
