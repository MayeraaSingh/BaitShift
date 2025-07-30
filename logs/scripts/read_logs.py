import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
import os

def initialize_firebase():
    # Adjust filename to your actual Firebase admin key JSON
    key_path = os.path.join(os.path.dirname(__file__), 'firebase-service-account.json')
    if not firebase_admin._apps:
        cred = credentials.Certificate(key_path)
        firebase_admin.initialize_app(cred)
    return firestore.client()

def fetch_collection(db, collection_name):
    docs = db.collection(collection_name).stream()
    return [doc.to_dict() for doc in docs]

def main():
    print("Initializing Firebase...")
    db = initialize_firebase()

    print("Fetching trap_logs collection from Firebase...")
    trap_logs = fetch_collection(db, 'trap_logs')
    print(f"Fetched {len(trap_logs)} trap_logs records.")

    print("Fetching trap_sessions collection from Firebase...")
    trap_sessions = fetch_collection(db, 'trap_sessions')
    print(f"Fetched {len(trap_sessions)} trap_sessions records.")

    # Normalize logs and sessions JSON structures to flat tables
    df_logs = pd.json_normalize(trap_logs)
    df_sessions = pd.json_normalize(trap_sessions)

    # Convert timestamps safely
    if 'timestamp_iso' in df_logs.columns:
        df_logs['timestampISO'] = pd.to_datetime(df_logs['timestamp_iso'], errors='coerce')
    else:
        # fallback in case of millisecond UNIX timestamp field
        df_logs['timestampISO'] = pd.to_datetime(df_logs['timestamp'], unit='ms', errors='coerce')

    # Message length fallback
    if 'message_length' in df_logs.columns:
        df_logs['msg_len'] = pd.to_numeric(df_logs['message_length'], errors='coerce').fillna(0).astype(int)
    else:
        # length of actual message string if numeric length not available
        df_logs['msg_len'] = df_logs['message'].fillna('').apply(len)

    # Construct attacker ID from IP hash and userAgent (hashes preserve privacy)
    df_logs['clientIP'] = df_logs.get('ip_address', df_logs.get('ip_address_hashed', 'unknown')).fillna('unknown')
    df_logs['userAgent'] = df_logs.get('user_agent', df_logs.get('userAgent', 'unknown')).fillna('unknown')
    df_logs['attacker_id'] = df_logs['clientIP'].astype(str) + "_" + df_logs['userAgent'].astype(str)

    # Delay since last message & copy-paste indicator
    df_logs['delay'] = pd.to_numeric(df_logs.get('delay', df_logs.get('delay_since_last_message', 0)), errors='coerce').fillna(0)
    df_logs['copy_paste'] = df_logs.get('copy_paste_indicator', 0).astype(int)

    # AI risk-related fields with safe defaults
    df_logs['ai_risk_score'] = pd.to_numeric(df_logs.get('ai_risk_score', -1), errors='coerce').fillna(-1)
    df_logs['ai_threat_category'] = df_logs.get('ai_threat_category', 'Unknown').fillna('Unknown')
    df_logs['ai_tone_label'] = df_logs.get('ai_tone_label', 'Unknown').fillna('Unknown')

    # Lure type
    df_logs['lure_type'] = df_logs.get('lure_type', 'unknown').fillna('unknown')

    # Merge Sessions for session metadata (optional, enhances analysis)
    if 'session_id' in df_sessions.columns:
        # Convert session timestamps
        df_sessions['session_start'] = pd.to_datetime(df_sessions['session_start'], unit='ms', errors='coerce')
        df_sessions['session_end'] = pd.to_datetime(df_sessions['session_end'], unit='ms', errors='coerce')
        df_sessions['session_duration'] = pd.to_numeric(df_sessions.get('session_duration', 0), errors='coerce').fillna(0)
        df_sessions['total_messages'] = pd.to_numeric(df_sessions.get('total_messages', 0), errors='coerce').fillna(0)

        session_cols = ['session_id', 'session_start', 'session_end', 'session_duration', 'total_messages']
        df_sessions = df_sessions[session_cols]

        # Join on session_id for enriched data
        df_logs = df_logs.merge(df_sessions, on='session_id', how='left')
    else:
        # Fallback in case no sessions are available
        df_logs['session_start'] = pd.NaT
        df_logs['session_end'] = pd.NaT
        df_logs['session_duration'] = 0
        df_logs['total_messages'] = 0

    # Sort by timestamp for time-based analyses downstream
    df_logs = df_logs.sort_values('timestampISO')

    # Save cleaned logs for clustering and dashboard
    out_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'trap_logs.csv')
    df_logs.to_csv(out_path, index=False)
    print(f"✅ Saved cleaned trap_logs.csv [{len(df_logs)} rows] at {out_path}")

if __name__ == '__main__':
    main()
