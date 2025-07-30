import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os
import altair as alt
import pydeck as pdk

##### Dashboard Configuration #####
st.set_page_config(
    page_title="BaitShift Threat Dashboard", 
    layout="wide",
    initial_sidebar_state="expanded"
)

##### Data Loading and Preparation #####
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data')
DATA_FILE = os.path.join(DATA_PATH, 'trap_logs_clustered.csv')

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE, low_memory=False)

    # Parse dates etc
    df['timestampISO'] = pd.to_datetime(df['timestampISO'], errors='coerce')
    df['geolocation.country'] = df.get('geolocation.country', 'Unknown').fillna('Unknown')
    df['copy_paste_like'] = df.get('copy_paste_like', df.get('copy_paste', 0)).astype(int)
    df['msg_len'] = df.get('msg_len', df.get('message_length', 0))
    df['delay'] = df.get('delay', df.get('delay_since_last_message', 0)).fillna(0)
    df['ai_risk_score'] = pd.to_numeric(df.get('ai_risk_score', -1), errors='coerce').fillna(-1)
    
    # Defensive for cluster_id column
    if 'cluster_id' not in df.columns:
        df['cluster_id'] = -1    
    if 'cluster_size' not in df.columns:
        df['cluster_size'] = 1
    
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

##### Key Performance Indicators (KPIs) #####
st.title("🕵️‍♂️ BaitShift Threat Intelligence Dashboard")
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
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Message Count")
ax.set_title("Hourly Attack Messages")
ax.set_xticks(range(24))
st.pyplot(fig)

##### Top Attacker Countries #####

st.subheader("Top Attacker Countries")
st.pyplot(fig)

country_counts = df_filtered['geolocation.country'].value_counts().head(10)
fig, ax = plt.subplots()
sns.barplot(
    x=country_counts.values,
    y=country_counts.index,
    hue=country_counts.index,  # Assign y variable to hue
    palette="Blues_d",
    ax=ax,
    legend=False
)
ax.set_xlabel("Message Count")
ax.set_ylabel("")
st.pyplot(fig)

st.markdown("---")

##### Cluster Size Distribution #####
st.subheader("Cluster Size Distribution")
cluster_sizes = df_filtered[df_filtered['cluster_id'] != -1]['cluster_id'].value_counts().sort_index()

if not cluster_sizes.empty:
    fig, ax = plt.subplots(figsize=(12, 4))
    cluster_sizes.plot(kind='bar', color='#fbbc05', ax=ax)
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("Number of Messages")
    ax.set_title("Messages per Cluster")
    ax.tick_params(axis='x', rotation=0)
    st.pyplot(fig)
else:
    st.info("No clusters data to show.")

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
        # ...existing code...
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
    st.write("Top AI Tone Labels in Cluster:")
    st.table(tone_counts)

##### Word Cloud: Cluster Messages #####
st.subheader("Word Cloud of Messages")
text_corpus = " ".join(cluster_df['message'].dropna().astype(str))
if text_corpus.strip():
    wc = WordCloud(width=800, height=300, background_color='white').generate(text_corpus)
    fig, ax = plt.subplots(figsize=(12,4))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)
else:
    st.info("No messages available to generate word cloud.")

##### Timeline: Messages by Attacker #####
st.subheader("Message Timeline by Attacker")
timeline_data = cluster_df[['timestampISO', 'attacker_id']].dropna()
timeline_data['timestamp_str'] = timeline_data['timestampISO'].dt.strftime('%Y-%m-%d %H:%M:%S')

chart = alt.Chart(timeline_data).mark_circle(size=70).encode(
    x='timestampISO:T',
    y=alt.Y('attacker_id:N', axis=alt.Axis(labels=False)),
    tooltip=['attacker_id', 'timestamp_str']
).properties(height=300, width=900)

st.altair_chart(chart, use_container_width=True)

##### Histograms: Message Length & Delay #####
st.subheader("Message Length and Delay Distributions")
colA, colB = st.columns(2)

with colA:
    fig, ax = plt.subplots()
    sns.histplot(cluster_df['msg_len'], bins=30, color='#2C8EAD', ax=ax)
    ax.set_xlabel("Message Length (chars)")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

with colB:
    fig, ax = plt.subplots()
    sns.histplot(cluster_df['delay'], bins=30, color='#F4777F', ax=ax)
    ax.set_xlabel("Delay Since Last Msg (ms)")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

##### Session Duration Distribution #####
st.subheader("Session Duration Distribution")
if 'session_duration' in cluster_df.columns and cluster_df['session_duration'].notna().any():
    fig, ax = plt.subplots(figsize=(10,3))
    sns.histplot(cluster_df['session_duration'].dropna(), bins=30, color='#54A24B', ax=ax)
    ax.set_xlabel("Session Duration (seconds)")
    ax.set_ylabel("Count")
    st.pyplot(fig)
else:
    st.info("Session duration data not available.")

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

        st.pydeck_chart(pdk.Deck(
            map_style='mapbox://styles/mapbox/light-v10',
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
            st.write(f"Top {col.split('.')[-1].capitalize()}s")
            st.bar_chart(cluster_df[col].value_counts().head(5))

st.markdown("---")
st.caption("BaitShift | AI-Powered Honeytrap Threat Intel | Developed by Person C")

