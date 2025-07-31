import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
import os
from user_agents import parse  # pip install pyyaml ua-parser user-agents

def enrich_user_agent_fields(df):
    browsers, oses, devices = [], [], []
    for ua_string in df["user_agent"].fillna(""):
        try:
            ua = parse(ua_string)
            browsers.append(ua.browser.family if ua.browser.family else "Unknown")
            oses.append(ua.os.family if ua.os.family else "Unknown")
            devices.append(ua.device.family if ua.device.family else "Unknown")
        except Exception:
            browsers.append("Unknown")
            oses.append("Unknown")
            devices.append("Unknown")
    df["user_agent_parsed.browser"] = browsers
    df["user_agent_parsed.os"] = oses
    df["user_agent_parsed.device"] = devices
    return df

def main():
    print("Initializing Firebase...")
    key_path = os.path.join(os.path.dirname(__file__), 'firebase-service-account.json')
    if not firebase_admin._apps:
        cred = credentials.Certificate(key_path)
        firebase_admin.initialize_app(cred)
    db = firestore.client()

    print("Fetching trap_logs...")
    trap_logs = [doc.to_dict() for doc in db.collection('trap_logs').stream()]
    print("Fetching trap_sessions...")
    trap_sessions = [doc.to_dict() for doc in db.collection('trap_sessions').stream()]

    df_logs = pd.json_normalize(trap_logs)
    df_sessions = pd.json_normalize(trap_sessions)

    # Timestamp
    if 'timestamp_iso' in df_logs.columns:
        df_logs['timestampISO'] = pd.to_datetime(df_logs['timestamp_iso'], errors='coerce')
    else:
        df_logs['timestampISO'] = pd.to_datetime(df_logs['timestamp'], unit='ms', errors='coerce')

    # Message length
    if 'message_length' in df_logs.columns:
        df_logs['msg_len'] = pd.to_numeric(df_logs['message_length'], errors='coerce').fillna(0).astype(int)
    else:
        df_logs['msg_len'] = df_logs['message'].fillna('').apply(len)

    # Construct attacker ID
    df_logs['clientIP'] = df_logs.get('ip_address', df_logs.get('ip_address_hashed', 'unknown')).fillna('unknown')
    df_logs['user_agent'] = df_logs.get('user_agent', df_logs.get('userAgent', 'unknown')).fillna('unknown')
    df_logs['attacker_id'] = df_logs['clientIP'].astype(str) + "_" + df_logs['user_agent'].astype(str)

    # Enrich User-Agent Parsed
    df_logs = enrich_user_agent_fields(df_logs)
    df_logs["user_agent_parsed.browser"] = df_logs["user_agent_parsed.browser"].fillna("Unknown")
    df_logs["user_agent_parsed.os"] = df_logs["user_agent_parsed.os"].fillna("Unknown")
    df_logs["user_agent_parsed.device"] = df_logs["user_agent_parsed.device"].fillna("Unknown")

    # Copy-paste indicator
    df_logs['copy_paste'] = df_logs.get('copy_paste_indicator', False).fillna(False).astype(int)

    # Risk-related fields
    df_logs['ai_risk_score'] = pd.to_numeric(df_logs.get('ai_risk_score', -1), errors='coerce').fillna(-1)
    df_logs['ai_threat_category'] = df_logs.get('ai_threat_category', df_logs.get('threat_category', 'Unknown')).fillna('Unknown')
    df_logs['ai_tone_label'] = df_logs.get('ai_tone_label', df_logs.get('tone_label', 'Unknown')).fillna('Unknown')

    # Encode tone label and threat category for clustering (factorized label)
    df_logs['tone_label_code'] = pd.factorize(df_logs['ai_tone_label'])[0]
    df_logs['threat_category_code'] = pd.factorize(df_logs['ai_threat_category'])[0]

    # Lure type
    df_logs['lure_type'] = df_logs.get('lure_type', 'unknown').fillna('unknown')

    # Delay field
    df_logs['delay'] = pd.to_numeric(df_logs.get('delay_since_last_message', 0), errors='coerce').fillna(0)

    # Session join and fields
    if 'session_id' in df_sessions.columns:
        if not pd.api.types.is_datetime64_any_dtype(df_sessions['session_start']):
            df_sessions['session_start'] = pd.to_datetime(df_sessions['session_start'], errors='coerce')
        if not pd.api.types.is_datetime64_any_dtype(df_sessions['session_end']):
            df_sessions['session_end'] = pd.to_datetime(df_sessions['session_end'], errors='coerce')
        df_sessions['session_duration'] = pd.to_numeric(df_sessions.get('session_duration', 0), errors='coerce').fillna(0)
        df_sessions['total_messages'] = pd.to_numeric(df_sessions.get('total_messages', 0), errors='coerce').fillna(0)
        session_cols = ['session_id', 'session_start', 'session_end', 'session_duration', 'total_messages']
        df_sessions = df_sessions[session_cols]
        df_logs = df_logs.merge(df_sessions, on='session_id', how='left')
    else:
        df_logs['session_start'] = pd.NaT
        df_logs['session_end'] = pd.NaT
        df_logs['session_duration'] = 0
        df_logs['total_messages'] = 0

    df_logs = df_logs.sort_values('timestampISO')
    out_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'trap_logs.csv')
    df_logs.to_csv(out_path, index=False)
    print(f"✅ Wrote cleaned logs with tone/threat codes at {out_path}")

if __name__ == '__main__':
    main()
