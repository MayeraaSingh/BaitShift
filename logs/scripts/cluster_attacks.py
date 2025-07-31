import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import os
import numpy as np

def main():
    input_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'trap_logs.csv')
    df = pd.read_csv(input_path)

    # Core Features
    features = [
        'msg_len',           # Message length
        'delay',             # Time between messages
        'copy_paste',        # Copy-paste indicator
        'ai_risk_score',     # Risk score (numeric)
        'tone_label_code',   # Encoded tone label
        'threat_category_code',  # Encoded threat category
        'session_duration',  # Session time
        'total_messages',    # Session message count
    ]
    # Defensive: ensure all features present and numeric
    for f in features:
        if f not in df.columns:
            df[f] = 0
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)

    # DBSCAN expects data scaled
    X = df[features].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # DBSCAN (tune eps/min_samples as needed for your data density)
    clustering = DBSCAN(eps=1.5, min_samples=2)
    df['cluster_id'] = clustering.fit_predict(X_scaled)
    df['cluster_size'] = df['cluster_id'].map(df['cluster_id'].value_counts())

    out_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'trap_logs_clustered.csv')
    df.to_csv(out_path, index=False)
    print(f"✅ Saved clustered logs with risk/tone/threat at {out_path}")

if __name__ == '__main__':
    main()
