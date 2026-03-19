# BaitShift 🎣

**BaitShift** is an AI-powered online predator detection system. It combines real-time threat analysis with interactive chat traps to detect, engage, and collect intelligence on scammers and online predators to prevent honeytrapping.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Installation & Setup](#installation--setup)
- [Running the Project](#running-the-project)
- [API Reference](#api-reference)
- [AI Model](#ai-model)
- [Configuration](#configuration)
- [Running Tests](#running-tests)

---

## Overview

BaitShift works in three phases:

1. **Detection** — A Chrome browser extension intercepts suspicious messages, analyzes them with an NLP server, and surfaces a risk score, tone classification, and a suggested bait reply.
2. **Engagement** — The user can paste the AI-generated bait reply to keep the scammer engaged, or lure them to a fake "secure chat" site (the trap site).
3. **Intelligence Collection** — The trap site silently logs behavioral signals (typing speed, copy-paste patterns, session duration), network data (IP, geolocation), and device fingerprints, all stored in Firebase Firestore for later analysis.

A Streamlit analytics dashboard visualizes the collected data, clusters attack campaigns, and exports intelligence reports.

---

## Features

- 🤖 **AI-Powered Threat Detection** — 91.7% accuracy tone classification using a fine-tuned DistilBERT model with an ensemble risk scorer.
- 🕵️ **Behavioral Analysis** — Tracks typing patterns, copy-paste behavior, and device/browser fingerprinting.
- 🌍 **Geolocation Tracking** — IP-based attacker location mapping using MaxMind GeoLite2 (with a free API fallback).
- 💬 **Dynamic Bait Reply Generation** — Uses the Google Gemini API and GPT-2 to generate context-aware replies.
- 📊 **Attack Dashboard** — Streamlit-based analytics with clustering, geoheatmaps, and CSV/PNG export.
- 🔗 **Multi-Component Integration** — Chrome extension + Flask backend + NLP server + Streamlit log dashboard.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    ATTACKER (SCAMMER)                             │
└─────────────────────┬────────────────────────────────────────────┘
                       │ Messages on WhatsApp Web / other chat
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│        CHROME BROWSER EXTENSION (Manifest V3)                     │
│  background.js  ·  content.js  ·  UI Panel                       │
└─────────────────────┬────────────────────────────────────────────┘
                       │ POST /analyze
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│        NLP ANALYSIS SERVER  (Flask · Port 5001)                   │
│  DistilBERT sentiment · keyword category · GPT-2 reply gen       │
└─────────────────────┬────────────────────────────────────────────┘
                       │
           ┌───────────┴───────────┐
           │                       │
           ▼                       ▼
   Copy bait reply         Navigate to trap site
   to WhatsApp             http://localhost:8000/securechat
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────┐
│        TRAP SITE — Fake Secure Chat  (Port 8000)                  │
│  frontend/index.html · frontend/script.js                        │
│  Collects: typing time · session data · IP · user agent          │
│                  │ POST /log, /session, /generate                 │
│                  ▼                                                │
│        BACKEND API  (Flask · Port 5000)                           │
│  app.py · ai_client.py · Gemini API integration                  │
└─────────────────────┬────────────────────────────────────────────┘
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
    Gemini API    Firebase       GeoIP
    (replies)   Firestore      (MaxMind /
                (storage)       ipapi.co)
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│        ANALYTICS DASHBOARD  (Streamlit · Port 8501)               │
│  logs/dashboard/app.py                                           │
│  Reads trap_logs_clustered.csv produced by cluster_attacks.py    │
└──────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. User sees suspicious message → Extension analyzes it via the NLP server.
2. Extension panel shows risk level, tone, and a suggested bait reply.
3. User pastes bait reply **or** sends the attacker a trap site link.
4. Fake chat site collects behavioral + network data, posts to backend.
5. Backend enriches with AI analysis and geolocation, stores in Firebase.
6. `cluster_attacks.py` groups messages into campaigns (writes CSV).
7. Streamlit dashboard visualizes patterns and exports intelligence.

---

## Project Structure

```
BaitShift/
├── extension/                   # Chrome browser extension (Manifest V3)
│   ├── manifest.json            # Permissions, content scripts, icons
│   ├── background.js            # Service worker – context menu registration
│   ├── content.js               # Intercepts selected text, renders panel
│   ├── style.css                # Extension panel styling
│   └── icons/                   # Extension icons
│
├── trap-site/
│   ├── frontend/                # Fake "secure chat" trap page
│   │   ├── index.html           # Chat UI
│   │   ├── script.js            # Session tracking + data collection
│   │   ├── server.py            # Simple HTTP server (port 8000)
│   │   ├── style.css
│   │   └── firebase.js          # Firebase SDK config
│   │
│   ├── backend/                 # Main Flask API (port 5000)
│   │   ├── app.py               # /log, /session, /generate endpoints
│   │   ├── ai_client.py         # AI inference wrapper
│   │   ├── requirements.txt
│   │   ├── test_ai_integration.py
│   │   └── test_complete_integration.py
│   │
│   └── ai_model/                # Threat detection AI
│       ├── main.py              # Public inference interface
│       ├── core_ai.py           # Ensemble models (RF + GBM + Ridge)
│       ├── trainer.py           # Training pipeline
│       ├── data_loader.py       # Dataset loading utilities
│       ├── dataset.json         # 240 labeled training samples
│       ├── models/              # Saved model artifacts (gitignored)
│       │   ├── tone_model/      # Fine-tuned DistilBERT
│       │   ├── tone_tokenizer/
│       │   ├── risk_ensemble_model.pkl
│       │   ├── risk_vectorizer.pkl
│       │   ├── risk_scaler.pkl
│       │   └── training_metadata.json
│       ├── requirements.txt
│       └── README.md            # Detailed AI documentation
│
├── nlp/                         # NLP analysis server (port 5001)
│   ├── server.py                # Flask app with /analyze and /health
│   ├── analyze_message.py       # Message analysis logic
│   └── requirements.txt
│
├── logs/                        # Analytics & dashboard
│   ├── dashboard/
│   │   └── app.py               # Streamlit dashboard
│   ├── scripts/
│   │   ├── cluster_attacks.py   # K-means clustering of attack logs
│   │   └── read_logs.py         # Firebase log reader
│   ├── data/
│   │   └── trap_logs_clustered.csv
│   └── requirements.txt
│
└── .gitignore
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Browser Extension | Chrome Manifest V3, JavaScript |
| Chat Frontend | HTML5, CSS3, Vanilla JavaScript |
| Backend API | Python 3.9+, Flask, Flask-CORS |
| NLP Server | Python 3.9+, Flask |
| AI / ML | DistilBERT (HuggingFace Transformers), PyTorch, scikit-learn (Random Forest, Gradient Boosting, Ridge) |
| LLM Integration | Google Gemini API, GPT-2 |
| Database | Firebase Firestore (Firebase Admin SDK) |
| Geolocation | MaxMind GeoLite2 (GeoIP2), ipapi.co (fallback) |
| Analytics | Streamlit, Pandas, NumPy, Matplotlib, Seaborn, Plotly, WordCloud, pydeck |

---

## Installation & Setup

### Prerequisites

- Python 3.9+
- Google Chrome (for the browser extension)
- A [Google Gemini API key](https://aistudio.google.com/)
- A Firebase project with a service account JSON file

---

### 1. Backend (Trap Site — Port 5000)

```bash
cd trap-site/backend
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in `trap-site/backend/`:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

Place your Firebase service account file in the same directory:

```
trap-site/backend/firebase-service-account.json
```

Optionally, add a MaxMind [GeoLite2-City.mmdb](https://dev.maxmind.com/geoip/geolite2-free-geolocation-data) database to the same directory (the backend automatically falls back to `ipapi.co` if the file is absent).

---

### 2. AI Model

```bash
cd trap-site/ai_model
pip install -r requirements.txt
# DistilBERT weights are downloaded automatically from HuggingFace on first use.
```

---

### 3. NLP Server (Port 5001)

```bash
cd nlp
pip install -r requirements.txt
```

---

### 4. Frontend Server (Port 8000)

No installation required — the trap site frontend is served by a plain Python HTTP server bundled in `trap-site/frontend/server.py`.

---

### 5. Analytics Dashboard (Port 8501)

```bash
cd logs
pip install -r requirements.txt
```

---

### 6. Chrome Extension

1. Open Chrome and navigate to `chrome://extensions/`.
2. Enable **Developer mode** (toggle in the top-right corner).
3. Click **Load unpacked** and select the `extension/` directory.

---

## Running the Project

Start each component in a separate terminal. The recommended order is:

```bash
# Terminal 1 — Backend API
cd trap-site/backend
source venv/bin/activate
python app.py
# → http://localhost:5000

# Terminal 2 — NLP Analysis Server
cd nlp
python server.py
# → http://localhost:5001

# Terminal 3 — Trap Site Frontend
cd trap-site/frontend
python server.py
# → http://localhost:8000  (chat UI at /securechat)

# Terminal 4 — Analytics Dashboard
cd logs
streamlit run dashboard/app.py
# → http://localhost:8501
```

### Using the Browser Extension

1. On any website (e.g. WhatsApp Web), select the suspicious text.
2. Right-click → **Scan this text with BaitShift**.
3. The extension panel shows the risk level, detected tone, threat category, and an AI-generated bait reply.
4. Copy the reply and paste it into the chat to keep the scammer engaged.

---

## API Reference

### Backend Flask API — Port 5000

| Endpoint | Method | Description |
|---|---|---|
| `/log` | POST | Log a message with AI analysis and attacker metadata |
| `/session` | POST | Update session summary (start/end time, message count) |
| `/generate` | POST | Generate an AI bait reply via the Gemini API |

#### `POST /log`

```json
// Request
{
  "message": "string",
  "sessionId": "string",
  "timestampISO": "string",
  "clientIP": "string",
  "typingTime": 1234,
  "userAgent": "string"
}

// Response
{ "success": true, "doc_id": "string" }
```

#### `POST /generate`

```json
// Request
{ "message": "string", "instruction": "string" }

// Response
{ "reply": "string" }
```

---

### NLP Server — Port 5001

| Endpoint | Method | Description |
|---|---|---|
| `/analyze` | POST | Analyze a message for risk, tone, and category |
| `/health` | GET | Health check |

#### `POST /analyze`

```json
// Request
{ "message": "hey beautiful, how old are you?" }

// Response
{
  "category": "Romance_Scam",
  "risk_level": "High",
  "reply": "string",
  "user_instructions": "string"
}
```

---

## AI Model

The threat detection pipeline in `trap-site/ai_model/` runs two stages:

```
Message Input
    │
    ├─ TONE CLASSIFICATION  (fine-tuned DistilBERT)
    │  6 classes: Friendly · Polite · Urgent · Technical · Manipulative · Threatening
    │
    ├─ FEATURE EXTRACTION
    │  Text stats · keyword groups · URL/email pattern detection
    │
    ├─ ENSEMBLE RISK SCORING
    │  TF-IDF (1500 features) → Random Forest + Gradient Boosting + Ridge → weighted vote
    │
    └─ OUTPUT
       {
         "risk_score": 0–100,
         "tone_label": "Manipulative",
         "confidence": 0.824,
         "threat_category": "Romance_Scam",
         "model_version": "improved_v2"
       }
```

**Threat categories**: `Romance_Scam`, `Tech_Support`, `Phishing`, `Financial`, `Threat`, `Info_Gathering`, `Unknown`, `Benign`

### Model Performance (v2.0)

| Metric | Value |
|---|---|
| Tone Accuracy | 91.7% |
| Tone F1 Score | 91.2% |
| Risk Score R² | 0.778 |
| Risk Score MAE | 12.3 points |
| Training Samples | 240 (augmented from 135) |
| Inference Speed | ~50 ms/message |

See [`trap-site/ai_model/README.md`](trap-site/ai_model/README.md) for full model documentation.

---

## Configuration

### Environment Variables

Create `trap-site/backend/.env`:

```env
# Required
GEMINI_API_KEY=your_gemini_api_key

# Firebase (from service account JSON)
FIREBASE_PROJECT_ID=your-project-id
FIREBASE_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----\n..."
FIREBASE_CLIENT_EMAIL=firebase-adminsdk@project.iam.gserviceaccount.com
FIREBASE_DB_URL=https://your-project.firebaseio.com

# Optional — uses ipapi.co as fallback when absent
GEOIP_DB_PATH=/path/to/GeoLite2-City.mmdb
```

### Files that must NOT be committed (already in `.gitignore`)

| File | Purpose |
|---|---|
| `firebase-service-account.json` | Firebase credentials |
| `.env` | API keys |
| `*.pkl`, `*.pth`, `*.bin` | Large model artifacts |
| `GeoLite2-City.mmdb` | Geolocation database |

---

## Running Tests

```bash
# AI integration test
cd trap-site/backend
python test_ai_integration.py

# Full end-to-end integration test
python test_complete_integration.py

# Quick smoke test for the AI model
cd trap-site/ai_model
python main.py

# Quick smoke test for the NLP analyzer
cd nlp
python -c "from analyze_message import analyze_message; print(analyze_message('hey beautiful how old are you?'))"

# Backend health check (requires running server)
curl http://localhost:5000/log \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"message":"test"}'
```
