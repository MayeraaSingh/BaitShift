import io
import zipfile
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os
import altair as alt
import pydeck as pdk

##### Dashboard Configuration #####
st.set_page_config(page_title="BaitShift Threat Dashboard", layout="wide", initial_sidebar_state="expanded")

##### Data Loading and Preparation #####
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data')
DATA_FILE = os.path.join(DATA_PATH, 'trap_logs_clustered.csv')

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE, low_memory=False)
    df['timestampISO'] = pd.to_datetime(df['timestampISO'], errors='coerce')
    df['geolocation.country'] = df.get('geolocation.country', 'Unknown').fillna('Unknown')
    df['copy_paste_like'] = df.get('copy_paste_like', df.get('copy_paste', 0)).astype(int)
    df['msg_len'] = df.get('msg_len', df.get('message_length', 0))
    df['delay'] = df.get('delay', df.get('delay_since_last_message', 0)).fillna(0)
    df['ai_risk_score'] = pd.to_numeric(df.get('ai_risk_score', -1), errors='coerce').fillna(-1)
    df['ai_tone_label'] = df.get('ai_tone_label', 'Unknown').fillna('Unknown')
    df['ai_threat_category'] = df.get('ai_threat_category', 'Unknown').fillna('Unknown')
    if 'cluster_id' not in df.columns: df['cluster_id'] = -1
    if 'cluster_size' not in df.columns: df['cluster_size'] = 1
    return df

df = load_data()

##### Sidebar: Data Filters #####
st.sidebar.header("🔎 Filter Threat Data")

min_date, max_date = df['timestampISO'].min().date(), df['timestampISO'].max().date()
selected_dates = st.sidebar.date_input("Date Range", (min_date, max_date), min_value=min_date, max_value=max_date)
df_filtered = df[(df['timestampISO'].dt.date >= selected_dates[0]) & (df['timestampISO'].dt.date <= selected_dates[1])]

all_clusters = sorted(df_filtered['cluster_id'].dropna().unique())
selected_clusters = st.sidebar.multiselect("Clusters", all_clusters, default=all_clusters)
df_filtered = df_filtered[df_filtered['cluster_id'].isin(selected_clusters)]


country_options = ["All"] + sorted(df_filtered['geolocation.country'].dropna().unique())
selected_country = st.sidebar.selectbox("Country", country_options)
if selected_country != "All":
    df_filtered = df_filtered[df_filtered['geolocation.country'] == selected_country]


if df_filtered.empty:
    st.warning("No data matches filters. Adjust filters please.")
    st.stop()

# --- ZIP Export: All Filtered Data + All Graphs ---
def export_dashboard_zip(selected_cluster_id=None):
    # Get cluster data if cluster ID is provided
    if selected_cluster_id is not None:
        cluster_df = df_filtered[df_filtered['cluster_id'] == selected_cluster_id]
    else:
        cluster_df = df_filtered  # Use all filtered data if no cluster selected

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        # 1. Add filtered data CSV
        zf.writestr("filtered_data.csv", df_filtered.to_csv(index=False))

        # 2. Hourly Attack Activity
        fig, ax = plt.subplots(figsize=(12, 3))
        hourly_counts = df_filtered.groupby(df_filtered['timestampISO'].dt.hour).size().reindex(range(24), fill_value=0)
        hourly_counts.plot(kind='bar', color='#4285f4', ax=ax)
        ax.set_xlabel("Hour of Day"); ax.set_ylabel("Message Count"); ax.set_title("Hourly Attack Messages")
        ax.set_xticks(range(24))
        buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
        zf.writestr("hourly_attack.png", buf.read())

        # 3. Top Attacker Countries
        fig, ax = plt.subplots()
        country_counts = df_filtered['geolocation.country'].value_counts().head(10)
        sns.barplot(x=country_counts.values, y=country_counts.index, palette="Blues_d", ax=ax)
        ax.set_xlabel("Message Count"); ax.set_ylabel("")
        buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
        zf.writestr("top_countries.png", buf.read())

        # 4. Cluster Size Distribution
        cluster_sizes = df_filtered[df_filtered['cluster_id'] != -1]['cluster_id'].value_counts().sort_index()
        if not cluster_sizes.empty:
            fig, ax = plt.subplots(figsize=(12, 4))
            cluster_sizes.plot(kind='bar', color='#fbbc05', ax=ax)
            ax.set_xlabel("Cluster ID"); ax.set_ylabel("Messages"); ax.set_title("Messages per Cluster")
            ax.tick_params(axis='x', rotation=0)
            buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
            zf.writestr("cluster_size_distribution.png", buf.read())

        # 5. High Risk Clusters Table
        high_risk_clusters = df_filtered[df_filtered['ai_risk_score'] >= 70]['cluster_id'].value_counts()
        if not high_risk_clusters.empty:
            hi_cluster_ids = high_risk_clusters.index.tolist()
            hi_df = df_filtered[df_filtered['cluster_id'].isin(hi_cluster_ids)]
            risk_summary = hi_df.groupby('cluster_id').agg(
                avg_risk = ('ai_risk_score', 'mean'),
                unique_attackers = ('attacker_id', 'nunique'),
                countries = ('geolocation.country', pd.Series.nunique),
                top_tone = ('ai_tone_label', pd.Series.mode),
                top_threat = ('ai_threat_category', pd.Series.mode),
                message_count = ('message', 'count'),
            ).sort_values('avg_risk', ascending=False)
            zf.writestr("high_risk_clusters.csv", risk_summary.to_csv())

        # 6. Cluster Drill-Down: Cluster Summary Table
        zf.writestr("selected_cluster_data.csv", cluster_df.to_csv(index=False))

        # 7. Word Cloud (as image, if available)
        text_corpus = " ".join(cluster_df['message'].dropna().astype(str))
        if text_corpus.strip():
            wc = WordCloud(width=800, height=300, background_color='white').generate(text_corpus)
            fig, ax = plt.subplots(figsize=(12,4)); ax.imshow(wc, interpolation='bilinear'); ax.axis("off")
            buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
            zf.writestr("wordcloud.png", buf.read())

        # 8. Timeline: Messages by Attacker (Altair chart as CSV data)
        timeline_data = cluster_df[['timestampISO', 'attacker_id']].dropna()
        timeline_data['timestamp_str'] = timeline_data['timestampISO'].dt.strftime('%Y-%m-%d %H:%M:%S')
        zf.writestr("timeline_data.csv", timeline_data.to_csv(index=False))

        # 9. Histograms: Message Length & Delay
        fig, ax = plt.subplots(); sns.histplot(cluster_df['msg_len'], bins=30, color='#2C8EAD', ax=ax)
        ax.set_xlabel("Message Length (chars)"); ax.set_ylabel("Frequency")
        buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
        zf.writestr("hist_msg_len.png", buf.read())

        fig, ax = plt.subplots(); sns.histplot(cluster_df['delay'], bins=30, color='#F4777F', ax=ax)
        ax.set_xlabel("Delay Since Last Msg (ms)"); ax.set_ylabel("Frequency")
        buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
        zf.writestr("hist_delay.png", buf.read())

        # 10. Session Duration Distribution
        if 'session_duration' in cluster_df.columns and cluster_df['session_duration'].notna().any():
            fig, ax = plt.subplots(figsize=(10,3))
            sns.histplot(cluster_df['session_duration'].dropna(), bins=30, color='#54A24B', ax=ax)
            ax.set_xlabel("Session Duration (seconds)"); ax.set_ylabel("Count")
            buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
            zf.writestr("session_duration.png", buf.read())

        # 11. Geographic Distribution Map (CSV of points)
        if {'geolocation.latitude', 'geolocation.longitude'}.issubset(cluster_df.columns):
            geo_pts = cluster_df[['geolocation.latitude', 'geolocation.longitude']].dropna()
            if not geo_pts.empty:
                geo_pts = geo_pts.rename(columns={'geolocation.latitude': 'lat', 'geolocation.longitude': 'lon'})
                zf.writestr("geo_points.csv", geo_pts.to_csv(index=False))

        # 12. Device, Browser, OS breakdowns (CSV)
        for col in ['user_agent_parsed.browser', 'user_agent_parsed.os', 'user_agent_parsed.device']:
            if col in cluster_df.columns:
                breakdown = cluster_df[col].fillna('Unknown').replace('', 'Unknown').value_counts().head(5)
                zf.writestr(f"top_{col.split('.')[-1]}s.csv", breakdown.to_csv())

    zip_buffer.seek(0)
    return zip_buffer

# Position the download button based on whether we're in cluster view
if 'cluster_choices' in locals():
    # We're in the cluster view, use the selected cluster
    st.download_button(
        label=f"⬇️ Download Data & Graphs for Cluster {select_cluster} (ZIP)",
        data=export_dashboard_zip(select_cluster),
        file_name=f"baitshift_cluster_{select_cluster}_export.zip",
        mime="application/zip"
    )
else:
    # We're in the main dashboard view
    st.download_button(
        label="⬇️ Download All Data & Graphs (ZIP)",
        data=export_dashboard_zip(),
        file_name="baitshift_dashboard_export.zip",
        mime="application/zip"
    )

##### Key Performance Indicators (KPIs) #####
st.title("🕵️ BaitShift Threat Intelligence Dashboard")
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
kpi1.metric("Messages", len(df_filtered))
kpi2.metric("Unique Attackers", df_filtered['attacker_id'].nunique())
kpi3.metric("Clusters", df_filtered['cluster_id'].nunique())
kpi4.metric("Countries", df_filtered['geolocation.country'].nunique())
kpi5.metric("High Risk Messages", (df_filtered['ai_risk_score'] >= 70).sum())
st.markdown("---")

##### Visualization: Attack Activity by Hour (UTC) #####
st.subheader("Attack Activity by Hour (UTC)")
df_filtered['hour'] = df_filtered['timestampISO'].dt.hour
hourly_counts = df_filtered.groupby('hour').size().reindex(range(24), fill_value=0)
fig, ax = plt.subplots(figsize=(12, 3))
hourly_counts.plot(kind='bar', color='#4285f4', ax=ax)
ax.set_xlabel("Hour of Day"); ax.set_ylabel("Message Count"); ax.set_title("Hourly Attack Messages")
ax.set_xticks(range(24))
st.pyplot(fig)

##### Top Attacker Countries #####
st.subheader("Top Attacker Countries")
country_counts = df_filtered['geolocation.country'].value_counts().head(10)
fig, ax = plt.subplots()
sns.barplot(x=country_counts.values, y=country_counts.index, palette="Blues_d", ax=ax)
ax.set_xlabel("Message Count"); ax.set_ylabel("")
st.pyplot(fig)
st.markdown("---")

##### Cluster Size Distribution #####
st.subheader("Cluster Size Distribution")
cluster_sizes = df_filtered[df_filtered['cluster_id'] != -1]['cluster_id'].value_counts().sort_index()
if not cluster_sizes.empty:
    fig, ax = plt.subplots(figsize=(12, 4))
    cluster_sizes.plot(kind='bar', color='#fbbc05', ax=ax)
    ax.set_xlabel("Cluster ID"); ax.set_ylabel("Messages"); ax.set_title("Messages per Cluster")
    ax.tick_params(axis='x', rotation=0)
    st.pyplot(fig)
else: st.info("No clusters data to show.")

##### Highlight: High-Risk Clusters #####
high_risk_clusters = df_filtered[df_filtered['ai_risk_score'] >= 70]['cluster_id'].value_counts()
if not high_risk_clusters.empty:
    st.subheader("🚩 High Risk Clusters (Avg AI Risk ≥ 70)")
    hi_cluster_ids = high_risk_clusters.index.tolist()
    hi_df = df_filtered[df_filtered['cluster_id'].isin(hi_cluster_ids)]
    risk_summary = hi_df.groupby('cluster_id').agg(
        avg_risk = ('ai_risk_score', 'mean'),
        unique_attackers = ('attacker_id', 'nunique'),
        countries = ('geolocation.country', pd.Series.nunique),
        top_tone = ('ai_tone_label', pd.Series.mode),
        top_threat = ('ai_threat_category', pd.Series.mode),
        message_count = ('message', 'count'),
    ).sort_values('avg_risk', ascending=False)
    st.dataframe(risk_summary.style.background_gradient(cmap="Reds", subset=['avg_risk']))

st.markdown("---")

##### Cluster Drill-Down: Explore Details #####
st.header("🔍 Explore a Cluster in Detail")
cluster_choices = sorted(df_filtered['cluster_id'].unique())
select_cluster = st.selectbox("Select Cluster", cluster_choices)
cluster_df = df_filtered[df_filtered['cluster_id'] == select_cluster]

st.markdown(f"### Cluster {select_cluster} Summary")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Messages", len(cluster_df))
col2.metric("Unique Attackers", cluster_df['attacker_id'].nunique())
col3.metric("Avg Session Duration (secs)", f"{cluster_df['session_duration'].mean():.1f}")
col4.metric("Avg Message Length (chars)", f"{cluster_df['msg_len'].mean():.1f}")
col5.metric("Copy-Paste %", f"{100*cluster_df['copy_paste_like'].mean():.2f}")
if 'ai_risk_score' in cluster_df.columns:
    avg_risk = cluster_df.loc[cluster_df['ai_risk_score'] >= 0, 'ai_risk_score'].mean()
    col2.metric("Avg AI Risk Score", f"{avg_risk:.2f}")
if 'ai_tone_label' in cluster_df.columns:
    tone_counts = cluster_df['ai_tone_label'].fillna('Unknown').value_counts().head(5)
    st.write("Top AI Tone Labels in Cluster:"); st.table(tone_counts)
if 'ai_threat_category' in cluster_df.columns:
    threat_counts = cluster_df['ai_threat_category'].fillna('Unknown').value_counts().head(5)
    st.write("Top Threat Categories in Cluster:"); st.table(threat_counts)

##### Word Cloud: Cluster Messages #####
st.subheader("Word Cloud of Messages")
text_corpus = " ".join(cluster_df['message'].dropna().astype(str))
if text_corpus.strip():
    wc = WordCloud(width=800, height=300, background_color='white').generate(text_corpus)
    fig, ax = plt.subplots(figsize=(12,4)); ax.imshow(wc, interpolation='bilinear'); ax.axis("off"); st.pyplot(fig)
else: st.info("No messages available to generate word cloud.")

##### Timeline: Messages by Attacker #####
st.subheader("Message Timeline by Attacker")
timeline_data = cluster_df[['timestampISO', 'attacker_id']].dropna()
timeline_data['timestamp_str'] = timeline_data['timestampISO'].dt.strftime('%Y-%m-%d %H:%M:%S')
chart = alt.Chart(timeline_data).mark_circle(size=70).encode(
    x='timestampISO:T', y=alt.Y('attacker_id:N', axis=alt.Axis(labels=False)),
    tooltip=['attacker_id', 'timestamp_str']
).properties(height=300, width=900)
st.altair_chart(chart, use_container_width=True)

##### Histograms: Message Length & Delay #####
st.subheader("Message Length and Delay Distributions")
colA, colB = st.columns(2)
with colA:
    fig, ax = plt.subplots(); sns.histplot(cluster_df['msg_len'], bins=30, color='#2C8EAD', ax=ax)
    ax.set_xlabel("Message Length (chars)"); ax.set_ylabel("Frequency"); st.pyplot(fig)
with colB:
    fig, ax = plt.subplots(); sns.histplot(cluster_df['delay'], bins=30, color='#F4777F', ax=ax)
    ax.set_xlabel("Delay Since Last Msg (ms)"); ax.set_ylabel("Frequency"); st.pyplot(fig)

##### Session Duration Distribution #####
st.subheader("Session Duration Distribution")
if 'session_duration' in cluster_df.columns and cluster_df['session_duration'].notna().any():
    fig, ax = plt.subplots(figsize=(10,3))
    sns.histplot(cluster_df['session_duration'].dropna(), bins=30, color='#54A24B', ax=ax)
    ax.set_xlabel("Session Duration (seconds)"); ax.set_ylabel("Count"); st.pyplot(fig)
else: st.info("Session duration data not available.")

##### Geographic Distribution Map (if available) #####
if {'geolocation.latitude', 'geolocation.longitude'}.issubset(cluster_df.columns):
    geo_pts = cluster_df[['geolocation.latitude', 'geolocation.longitude']].dropna()
    if not geo_pts.empty:
        st.subheader("Attacker Geo Distribution (Map)")
        geo_pts = geo_pts.rename(columns={'geolocation.latitude': 'lat', 'geolocation.longitude': 'lon'})

        # Define a function to scale radius based on zoom
        def zoom_to_radius(zoom):
            # Example: radius halves for each zoom level increase
            base_radius = 10000  # at zoom=2
            return base_radius / (2 ** (zoom - 2))

        # Set initial zoom
        initial_zoom = 2
        dynamic_radius = zoom_to_radius(initial_zoom)

        # Try different map styles for better compatibility
        map_styles = [
            'https://basemaps.cartocdn.com/gl/positron-gl-style/style.json',  # Carto Light
            'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json',  # Carto Dark
            'https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json',  # Carto Voyager
            None  # Default PyDeck style
        ]
        
        map_style = map_styles[0]  # Use Carto Light as default
        
        st.pydeck_chart(pdk.Deck(
            map_style=map_style,
            initial_view_state=pdk.ViewState(
                latitude=geo_pts['lat'].mean(),
                longitude=geo_pts['lon'].mean(),
                zoom=initial_zoom,
                pitch=0,
            ),
            layers=[
                pdk.Layer(
                    'ScatterplotLayer',
                    data=geo_pts,
                    get_position='[lon, lat]',
                    get_radius=dynamic_radius,
                    get_fill_color=[255, 80, 80, 255],
                    pickable=True,
                ),
            ],
        ))
    else:
        st.info("Geo location data not available for this cluster.")
else:
    st.info("Geo location coords missing in dataset.")


##### Device, Browser, and OS Breakdown #####
with st.expander("Device, Browser, OS Summary"):
    for col in ['user_agent_parsed.browser', 'user_agent_parsed.os', 'user_agent_parsed.device']:
        if col in cluster_df.columns:
            cluster_df[col] = cluster_df[col].fillna('Unknown').replace('', 'Unknown')
            st.write(f"Top {col.split('.')[-1].capitalize()}s")
            st.bar_chart(cluster_df[col].value_counts().head(5))

st.markdown("---")
st.caption("BaitShift | AI-Powered Honeytrap Threat Intel")
