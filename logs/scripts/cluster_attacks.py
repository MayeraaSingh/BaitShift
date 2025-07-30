import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import os
import numpy as np

def main():
    input_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'trap_logs.csv')
    df = pd.read_csv(input_path)

    # Ensure numeric and no-NaN for clustering features
    df['msg_len'] = pd.to_numeric(df.get('msg_len', 0), errors='coerce').fillna(0)
    df['delay'] = pd.to_numeric(df.get('delay', 0), errors='coerce').fillna(0)

    # Extract hour from timestamp for time-of-day pattern
    df['hour'] = pd.to_datetime(df['timestampISO'], errors='coerce').dt.hour.fillna(0).astype(int)

    # Binary copy-paste indicator
    df['copy_paste'] = df.get('copy_paste', 0).astype(int)

    df['session_duration'] = pd.to_numeric(df.get('session_duration', 0), errors='coerce').fillna(0)
    df['total_messages'] = pd.to_numeric(df.get('total_messages', 0), errors='coerce').fillna(0)

    # For AI risk score, replace missing -1 with median of available
    median_risk = df.loc[df['ai_risk_score'] >= 0, 'ai_risk_score'].median()
    median_risk = median_risk if not np.isnan(median_risk) else 50  # Default median fallback
    df['risk_for_clustering'] = df['ai_risk_score'].apply(lambda x: median_risk if x < 0 else x)

    features = ['msg_len', 'delay', 'hour', 'copy_paste', 'session_duration', 'total_messages', 'risk_for_clustering']

    # Ensure no missing values in features
    X = df[features].fillna(0).values

    # Feature scaling for DBSCAN
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # DBSCAN clustering with tuned eps and min_samples.
    clustering = DBSCAN(eps=1.4, min_samples=2)
    df['cluster_id'] = clustering.fit_predict(X_scaled)

    # Calculate cluster sizes
    cluster_counts = df['cluster_id'].value_counts()
    df['cluster_size'] = df['cluster_id'].map(cluster_counts)

    out_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'trap_logs_clustered.csv')
    df.to_csv(out_path, index=False)
    print(f"✅ Saved clustered data at {out_path} with {len(df)} rows and clusters assigned.")

if __name__ == '__main__':
    main()
