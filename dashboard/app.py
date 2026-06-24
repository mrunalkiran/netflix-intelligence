import base64
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from pipeline.analyse import (
    load_clean, content_type_split, top_genres,
    content_per_year, top_countries, ratings_distribution,
    top_directors
)
from rag.query import ask

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Netflix Intelligence",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Import font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Remove top padding */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
    }
    [data-testid="stAppViewContainer"] > div:first-child {
        padding-top: 0 !important;
    }

    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Cinematic background */
    .stApp {
        background-color: #0a0a0a;
        background-image:
            linear-gradient(rgba(10,10,10,0.92), rgba(10,10,10,0.92)),
            repeating-linear-gradient(
                0deg,
                transparent,
                transparent 120px,
                rgba(229,9,20,0.03) 120px,
                rgba(229,9,20,0.03) 121px
            ),
            repeating-linear-gradient(
                90deg,
                transparent,
                transparent 120px,
                rgba(229,9,20,0.03) 120px,
                rgba(229,9,20,0.03) 121px
            );
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #111111;
        border-right: 1px solid #222;
    }

    /* Hero banner */
    .hero {
        background: linear-gradient(135deg, #1a0000 0%, #0a0a0a 50%, #1a0a00 100%);
        border: 1px solid #E50914;
        border-radius: 16px;
        padding: 40px;
        margin-bottom: 32px;
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(229,9,20,0.15) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-title {
        font-size: 2.8rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0;
        line-height: 1.2;
    }
    .hero-title span {
        color: #E50914;
    }
    .hero-subtitle {
        font-size: 1rem;
        color: #888;
        margin-top: 8px;
        font-weight: 400;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(229,9,20,0.15);
        border: 1px solid rgba(229,9,20,0.4);
        color: #E50914;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 16px;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    /* Metric cards */
    .metric-card {
        background: #111111;
        border: 1px solid #222;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        transition: border-color 0.2s;
    }
    .metric-card:hover {
        border-color: #E50914;
    }
    .metric-icon {
        font-size: 1.8rem;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
        line-height: 1;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #666;
        margin-top: 6px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 500;
    }

    /* Section headers */
    .section-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .section-header::after {
        content: '';
        flex: 1;
        height: 1px;
        background: #222;
        margin-left: 12px;
    }

    /* Insight box */
    .insight-box {
        background: linear-gradient(135deg, #1a0a00, #0a0a0a);
        border: 1px solid rgba(229,9,20,0.3);
        border-left: 3px solid #E50914;
        border-radius: 8px;
        padding: 16px 20px;
        margin-top: 16px;
        color: #ccc;
        font-size: 0.9rem;
    }

    /* Chat messages */
    .chat-message-user {
        background: #1a1a1a;
        border: 1px solid #333;
        border-radius: 12px 12px 4px 12px;
        padding: 14px 18px;
        margin: 8px 0;
        color: #fff;
        font-size: 0.95rem;
        text-align: right;
    }
    .chat-message-ai {
        background: linear-gradient(135deg, #1a0000, #111);
        border: 1px solid rgba(229,9,20,0.25);
        border-radius: 12px 12px 12px 4px;
        padding: 14px 18px;
        margin: 8px 0;
        color: #ddd;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    .ai-label {
        font-size: 0.7rem;
        color: #E50914;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 6px;
    }

    /* Suggestion chips */
    .stButton button {
        background: #1a1a1a !important;
        border: 1px solid #333 !important;
        color: #ccc !important;
        border-radius: 20px !important;
        font-size: 0.8rem !important;
        padding: 4px 14px !important;
        transition: all 0.2s !important;
        width: 100%;
    }
    .stButton button:hover {
        border-color: #E50914 !important;
        color: #fff !important;
        background: #1a0000 !important;
    }

    /* Primary button */
    .stButton.primary button {
        background: #E50914 !important;
        border: none !important;
        color: white !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: transparent;
        gap: 4px;
        border-bottom: 1px solid #222;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #666;
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
        font-weight: 500;
        font-size: 0.9rem;
    }
    .stTabs [aria-selected="true"] {
        background: transparent !important;
        color: #E50914 !important;
        border-bottom: 2px solid #E50914 !important;
    }

    /* Source pills */
    .source-pill {
        display: inline-block;
        background: #1a1a1a;
        border: 1px solid #333;
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.75rem;
        color: #888;
        margin: 4px 4px 4px 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def get_data():
    return load_clean()

df = get_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    _logo_path = os.path.join(os.path.dirname(__file__), '..', 'images', 'netflix_logo.png')
    with open(_logo_path, 'rb') as _f:
        _logo_b64 = base64.b64encode(_f.read()).decode()
    st.markdown(f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&display=swap');
        </style>
        <div style='display:flex;align-items:center;gap:4px;margin-bottom:8px'>
            <img src='data:image/png;base64,{_logo_b64}' style='width:105px'>
            <span style='
                color:#ffffff;
                font-size:2.5rem;
                font-weight:400;
                font-family:"Bebas Neue",Impact,sans-serif;
                letter-spacing:0.02em;
                line-height:1.8;
                margin-top:4    px;
            '>IP</span>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    st.markdown("<div style='font-size:0.75rem;color:#555;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px'>Filters</div>", unsafe_allow_html=True)

    content_type = st.multiselect(
        "Content Type",
        options=df['type'].unique(),
        default=df['type'].unique(),
        label_visibility="collapsed"
    )

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.75rem;color:#555;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px'>Year Range</div>", unsafe_allow_html=True)

    years = df['year_added'].dropna().astype(int)
    year_range = st.slider(
        "Year Range",
        min_value=int(years.min()),
        max_value=int(years.max()),
        value=(2015, int(years.max())),
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown(f"""
        <div style='font-size:0.75rem;color:#444;line-height:1.8'>
            <div>📊 Dataset: Netflix Titles</div>
            <div>🤖 Model: GPT-3.5-turbo</div>
            <div>🔍 Vector DB: ChromaDB</div>
            <div>⚡ Built with Streamlit</div>
        </div>
    """, unsafe_allow_html=True)

# ── Filter data ───────────────────────────────────────────────────────────────
df_f = df[
    df['type'].isin(content_type) &
    (df['year_added'] >= year_range[0]) &
    (df['year_added'] <= year_range[1])
]

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
    <div class="hero">
        <div class="hero-badge">✦ RAG Powered Analytics</div>
        <div class="hero-title">Netflix <span>Intelligence</span> Platform</div>
        <div class="hero-subtitle">Explore 8,800+ titles with AI-powered insights and natural language querying</div>
    </div>
""", unsafe_allow_html=True)

# ── Metric cards ──────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
metrics = [
    ("🎬", f"{len(df_f):,}", "Total Titles", c1),
    ("🎥", f"{len(df_f[df_f['type']=='Movie']):,}", "Movies", c2),
    ("📺", f"{len(df_f[df_f['type']=='TV Show']):,}", "TV Shows", c3),
    ("🌍", f"{df_f['country'].nunique():,}", "Countries", c4),
]
for icon, val, label, col in metrics:
    col.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">{icon}</div>
            <div class="metric-value">{val}</div>
            <div class="metric-label">{label}</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["🎭  Genres", "🌍  Countries", "📈  Growth", "🎬  Directors", "🔍  Explorer", "🍿  Recommender", "🤖  Ask AI"])

# Plotly dark template
TEMPLATE = "plotly_dark"
NETFLIX_RED = "#E50914"
COLORS = ["#E50914", "#ff4d4d", "#ff8080", "#ffb3b3", "#ffe0e0"]

# ── Tab 1: Genres ─────────────────────────────────────────────────────────────
with tab1:
    col1, col2 = st.columns([3, 2], gap="large")

    with col1:
        st.markdown('<div class="section-header">🎭 Top Genres by Title Count</div>', unsafe_allow_html=True)
        genres = top_genres(df_f)
        fig = go.Figure(go.Bar(
            x=genres['count'],
            y=genres['listed_in'],
            orientation='h',
            marker=dict(
                color=genres['count'],
                colorscale=[[0, '#3d0000'], [1, '#E50914']],
                showscale=False
            ),
            text=genres['count'],
            textposition='outside',
            textfont=dict(color='#888', size=11)
        ))
        fig.update_layout(
            template=TEMPLATE,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=620,
            margin=dict(l=0, r=40, t=10, b=0),
            yaxis=dict(categoryorder='total ascending', gridcolor='#1a1a1a', color='#666'),
            xaxis=dict(gridcolor='#1a1a1a', color='#444'),
            showlegend=False
        )
        st.plotly_chart(fig, width='stretch')

    with col2:
        st.markdown('<div class="section-header">🍩 Content Split</div>', unsafe_allow_html=True)
        type_split = content_type_split(df_f)
        fig2 = go.Figure(go.Pie(
            labels=type_split['type'],
            values=type_split['count'],
            hole=0.65,
            marker=dict(colors=['#E50914', '#333333']),
            textinfo='label+percent',
            textfont=dict(color='white', size=13),
        ))
        fig2.update_layout(
            template=TEMPLATE,
            paper_bgcolor='rgba(0,0,0,0)',
            height=300,
            margin=dict(l=0, r=0, t=10, b=0),
            showlegend=False,
            annotations=[dict(
                text=f"{len(df_f):,}<br><span style='font-size:10px'>titles</span>",
                x=0.5, y=0.5, font_size=18, showarrow=False,
                font_color='white'
            )]
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<div class="section-header">⭐ Top Ratings</div>', unsafe_allow_html=True)
        ratings = ratings_distribution(df_f)
        ratings = ratings[ratings['rating'].str.len() < 10].head(6)
        fig3 = go.Figure(go.Bar(
            x=ratings['rating'],
            y=ratings['count'],
            marker_color=NETFLIX_RED,
            marker_opacity=0.8
        ))
        fig3.update_layout(
            template=TEMPLATE,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=220,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(gridcolor='#1a1a1a', color='#666'),
            yaxis=dict(gridcolor='#1a1a1a', color='#444'),
        )
        st.plotly_chart(fig3, use_container_width=True)

    # Insight
    top_genre = genres.iloc[-1]
    st.markdown(f"""
        <div class="insight-box">
            💡 <strong>{top_genre['listed_in']}</strong> dominates Netflix with
            <strong>{top_genre['count']:,} titles</strong> —
            making it the #1 genre on the platform.
        </div>
    """, unsafe_allow_html=True)

# ── Tab 2: Countries ──────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header">🌍 Top Content Producing Countries</div>', unsafe_allow_html=True)
    countries = top_countries(df_f)

    col1, col2 = st.columns([2, 3], gap="large")
    with col1:
        fig = go.Figure(go.Bar(
            x=countries['count'],
            y=countries['country'],
            orientation='h',
            marker=dict(
                color=countries['count'],
                colorscale=[[0, '#1a0000'], [1, '#E50914']],
                showscale=False
            ),
            text=countries['count'],
            textposition='outside',
            textfont=dict(color='#888', size=11)
        ))
        fig.update_layout(
            template=TEMPLATE,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=400,
            margin=dict(l=0, r=40, t=10, b=0),
            yaxis=dict(categoryorder='total ascending', gridcolor='#1a1a1a', color='#666'),
            xaxis=dict(gridcolor='#1a1a1a', color='#444'),
        )
        st.plotly_chart(fig, width='stretch')

    with col2:
        fig2 = go.Figure(go.Choropleth(
            locations=countries['country'],
            locationmode='country names',
            z=countries['count'],
            colorscale=[[0, '#1a0000'], [0.5, '#800000'], [1, '#E50914']],
            showscale=True,
            colorbar=dict(
                bgcolor='rgba(0,0,0,0)',
                tickcolor='#555',
                outlinecolor='#222'
            )
        ))
        fig2.update_layout(
            template=TEMPLATE,
            paper_bgcolor='rgba(0,0,0,0)',
            height=400,
            geo=dict(
                bgcolor='rgba(0,0,0,0)',
                lakecolor='rgba(0,0,0,0)',
                landcolor='#1a1a1a',
                showland=True,
                showocean=True,
                oceancolor='#0d0d0d',
                showlakes=False,
                showframe=False,
                coastlinecolor='#333',
            ),
            margin=dict(l=0, r=0, t=10, b=0)
        )
        st.plotly_chart(fig2, use_container_width=True)

    top_country = countries.iloc[0]
    st.markdown(f"""
        <div class="insight-box">
            💡 <strong>{top_country['country']}</strong> leads all nations with
            <strong>{top_country['count']:,} titles</strong> on Netflix —
            {round(top_country['count']/len(df_f)*100, 1)}% of the entire catalogue.
        </div>
    """, unsafe_allow_html=True)

# ── Tab 3: Growth ─────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">📈 Netflix Content Growth Over Time</div>', unsafe_allow_html=True)
    per_year = content_per_year(df_f)
    per_year = per_year[per_year['year_added'] >= 2010]

    # Calculate year-over-year growth %
    per_year['pct_change'] = per_year['count'].pct_change() * 100
    per_year['pct_change'] = per_year['pct_change'].fillna(0).round(1)

    # Best genre per year
    def best_genre_for_year(year):
        subset = df_f[df_f['year_added'] == year]
        if subset.empty:
            return 'N/A'
        genres = subset['listed_in'].str.split(',').explode().str.strip()
        return genres.value_counts().idxmax() if not genres.empty else 'N/A'

    per_year['top_genre'] = per_year['year_added'].apply(best_genre_for_year)

    # Build custom hover text
    per_year['hover_text'] = per_year.apply(lambda row: (
        f"<b>{int(row['year_added'])}</b><br>"
        f"📦 Titles Added: <b>{int(row['count']):,}</b><br>"
        f"📈 Growth: <b>{'↑' if row['pct_change'] >= 0 else '↓'} {abs(row['pct_change'])}%</b><br>"
        f"🎭 Top Genre: <b>{row['top_genre']}</b>"
    ), axis=1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=per_year['year_added'],
        y=per_year['count'],
        fill='tozeroy',
        fillcolor='rgba(229,9,20,0.15)',
        line=dict(color=NETFLIX_RED, width=3),
        mode='lines+markers',
        marker=dict(size=10, color=NETFLIX_RED, line=dict(color='white', width=2)),
        hovertemplate='%{customdata}<extra></extra>',
        customdata=per_year['hover_text']
    ))

    fig.update_layout(
        template=TEMPLATE,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=380,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(gridcolor='#1a1a1a', color='#666', tickmode='linear', dtick=1),
        yaxis=dict(gridcolor='#1a1a1a', color='#444'),
        hovermode='closest',
        hoverlabel=dict(
            bgcolor='#1a0000',
            bordercolor='#E50914',
            font=dict(color='white', size=13, family='Inter'),
        ),
    )
    st.plotly_chart(fig, width='stretch')

    # Stats row
    peak = per_year.loc[per_year['count'].idxmax()]
    total = per_year['count'].sum()
    avg = per_year['count'].mean()

    c1, c2, c3 = st.columns(3)
    c1.markdown(f"""<div class="metric-card">
        <div class="metric-icon">🏆</div>
        <div class="metric-value">{int(peak['year_added'])}</div>
        <div class="metric-label">Peak Year</div>
    </div>""", unsafe_allow_html=True)
    c2.markdown(f"""<div class="metric-card">
        <div class="metric-icon">📦</div>
        <div class="metric-value">{int(peak['count']):,}</div>
        <div class="metric-label">Titles in Peak Year</div>
    </div>""", unsafe_allow_html=True)
    c3.markdown(f"""<div class="metric-card">
        <div class="metric-icon">📊</div>
        <div class="metric-value">{int(avg):,}</div>
        <div class="metric-label">Avg Titles Per Year</div>
    </div>""", unsafe_allow_html=True)

# ── Tab 4: Directors ──────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-header">🎬 Top Directors on Netflix</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2], gap="large")

    with col1:
        directors = top_directors(df_f, top_n=15)
        fig = go.Figure(go.Bar(
            x=directors['director'],
            y=directors['count'],
            marker=dict(
                color=directors['count'],
                colorscale=[[0, '#3d0000'], [1, '#E50914']],
                showscale=False
            ),
            text=directors['count'],
            textposition='outside',
            textfont=dict(color='#888', size=11)
        ))
        fig.update_layout(
            template=TEMPLATE,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=420,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(gridcolor='#1a1a1a', color='#666', tickangle=-35),
            yaxis=dict(gridcolor='#1a1a1a', color='#444', title='Number of Titles'),
            showlegend=False,
            hoverlabel=dict(
                bgcolor='#1a0000',
                bordercolor='#E50914',
                font=dict(color='white', size=13)
            )
        )
        st.plotly_chart(fig, width='stretch')

    with col2:
        st.markdown('<div class="section-header">📋 Director Details</div>', unsafe_allow_html=True)
        directors_detail = df_f[df_f['director'] != 'Unknown'].groupby('director').agg(
            titles=('title', 'count'),
            types=('type', lambda x: ', '.join(x.unique())),
            genres=('listed_in', lambda x: x.value_counts().idxmax()),
            latest=('release_year', 'max')
        ).reset_index().sort_values('titles', ascending=False).head(15)

        # Build all cards as one HTML block inside a scrollable div
        cards_html = "<div style='height:420px;overflow-y:auto;padding-right:8px;scrollbar-width:thin;scrollbar-color:#E50914 #1a1a1a;'>"
        for _, row in directors_detail.iterrows():
            cards_html += f"""
                <div style='background:#111;border:1px solid #222;border-radius:10px;
                            padding:14px;margin-bottom:10px'>
                    <div style='font-weight:600;color:#fff;font-size:0.95rem'>{row['director']}</div>
                    <div style='color:#555;font-size:0.75rem;margin-top:6px;display:flex;gap:12px'>
                        <span>🎬 {row['titles']} titles</span>
                        <span>📺 {row['types']}</span>
                        <span>📅 Latest: {int(row['latest'])}</span>
                    </div>
                    <div style='color:#444;font-size:0.72rem;margin-top:4px'>
                        🎭 {row['genres'][:40]}...
                    </div>
                </div>
            """
        cards_html += "</div>"
        components.html(f"""
            <style>
                body {{ margin: 0; background: transparent; font-family: Inter, sans-serif; }}
                ::-webkit-scrollbar {{ width: 4px; }}
                ::-webkit-scrollbar-track {{ background: #1a1a1a; }}
                ::-webkit-scrollbar-thumb {{ background: #E50914; border-radius: 4px; }}
            </style>
            {cards_html}
        """, height=430, scrolling=False)

    top_dir = directors_detail.iloc[0]
    st.markdown(f"""
        <div class="insight-box">
            💡 <strong>{top_dir['director']}</strong> is Netflix's most prolific director
            with <strong>{top_dir['titles']} titles</strong> —
            primarily known for <strong>{top_dir['genres'][:30]}</strong> content.
        </div>
    """, unsafe_allow_html=True)

# ── Tab 5: Content Explorer ───────────────────────────────────────────────────
with tab5:
    st.markdown('<div class="section-header">🔍 Content Explorer</div>', unsafe_allow_html=True)

    # ── Filters row ───────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        selected_type = st.selectbox(
            "Type",
            options=["All", "Movie", "TV Show"]
        )
    with col2:
        all_genres = sorted(set(
            g.strip()
            for genres in df['listed_in'].dropna()
            for g in genres.split(',')
        ))
        selected_genre = st.selectbox("Genre", ["All"] + all_genres)

    with col3:
        all_ratings = sorted(df['rating'].dropna().unique().tolist())
        selected_rating = st.selectbox("Rating", ["All"] + all_ratings)

    with col4:
        all_countries = sorted(set(
            c.strip()
            for countries in df['country'].dropna()
            for c in countries.split(',')
            if c.strip() != 'Unknown'
        ))
        selected_country = st.selectbox("Country", ["All"] + all_countries)

    # ── Apply filters ─────────────────────────────────────────────────────────
    explored = df.copy()
    if selected_type != "All":
        explored = explored[explored['type'] == selected_type]
    if selected_genre != "All":
        explored = explored[explored['listed_in'].str.contains(selected_genre, na=False)]
    if selected_rating != "All":
        explored = explored[explored['rating'] == selected_rating]
    if selected_country != "All":
        explored = explored[explored['country'].str.contains(selected_country, na=False)]

    # ── Results count ─────────────────────────────────────────────────────────
    st.markdown(f"""
        <div style='margin:16px 0;font-size:0.85rem;color:#555'>
            Showing <span style='color:#E50914;font-weight:600'>{len(explored)}</span> titles
        </div>
    """, unsafe_allow_html=True)

    # ── Cards grid ────────────────────────────────────────────────────────────
    if len(explored) == 0:
        st.markdown("""
            <div style='text-align:center;padding:60px;color:#444'>
                😕 No titles found with these filters
            </div>
        """, unsafe_allow_html=True)
    else:
        # Show max 50 results
        results = explored.head(50)

        # Build cards HTML
        cards = ""
        for _, row in results.iterrows():
            content_type_icon = "🎥" if row['type'] == "Movie" else "📺"
            duration = str(row['duration']) if pd.notna(row['duration']) else "N/A"
            country = str(row['country']).split(',')[0].strip() if pd.notna(row['country']) else "Unknown"
            genres = str(row['listed_in'])[:35] if pd.notna(row['listed_in']) else "N/A"
            description = str(row['description'])[:120] + "..." if pd.notna(row['description']) else "No description available."
            year = int(row['release_year']) if pd.notna(row['release_year']) else "N/A"
            rating = str(row['rating']) if pd.notna(row['rating']) else "N/A"
            title = str(row['title'])

            cards += f"""
                <div style='background:#111;border:1px solid #222;border-radius:12px;
                            padding:18px;margin-bottom:12px;
                            transition:border-color 0.2s,transform 0.2s'
                     onmouseover="this.style.borderColor='#E50914';this.style.transform='translateX(4px)'"
                     onmouseout="this.style.borderColor='#222';this.style.transform='translateX(0)'">

                    <div style='display:flex;justify-content:space-between;align-items:flex-start'>
                        <div style='flex:1'>
                            <div style='font-size:1rem;font-weight:600;color:#fff'>
                                {content_type_icon} {title}
                            </div>
                            <div style='font-size:0.75rem;color:#555;margin-top:4px'>
                                📅 {year} &nbsp;|&nbsp; ⏱ {duration} &nbsp;|&nbsp; 🌍 {country}
                            </div>
                        </div>
                        <div style='display:flex;gap:6px;flex-shrink:0;margin-left:12px'>
                            <span style='background:rgba(229,9,20,0.15);border:1px solid rgba(229,9,20,0.3);
                                         color:#E50914;padding:3px 10px;border-radius:20px;font-size:0.7rem;
                                         font-weight:600'>{rating}</span>
                        </div>
                    </div>

                    <div style='font-size:0.78rem;color:#444;margin-top:8px;line-height:1.5'>
                        {description}
                    </div>

                    <div style='font-size:0.7rem;color:#333;margin-top:8px'>
                        🎭 {genres}
                    </div>
                </div>
            """

        import streamlit.components.v1 as components
        components.html(f"""
            <style>
                body {{ margin:0; background:transparent; font-family:Inter,sans-serif; }}
                ::-webkit-scrollbar {{ width:4px; }}
                ::-webkit-scrollbar-track {{ background:#1a1a1a; }}
                ::-webkit-scrollbar-thumb {{ background:#E50914; border-radius:4px; }}
            </style>
            <div style='height:600px;overflow-y:auto;padding-right:8px'>
                {cards}
            </div>
        """, height=620, scrolling=False)

# ── Tab 6: Recommender ────────────────────────────────────────────────────────
with tab6:
    st.markdown("""
        <div style='margin-bottom:24px'>
            <div style='font-size:1.3rem;font-weight:600;color:#fff;margin-bottom:6px'>
                🍿 AI Content Recommender
            </div>
            <div style='font-size:0.85rem;color:#555'>
                Tell us what you liked and we'll find similar titles from Netflix's catalogue
            </div>
        </div>
    """, unsafe_allow_html=True)

    # ── Input section ─────────────────────────────────────────────────────────
    col1, col2 = st.columns([3, 1], gap="large")

    with col1:
        liked_title = st.text_input(
            "Title you liked:",
            placeholder="e.g. Squid Game, Breaking Bad, The Crown...",
            label_visibility="collapsed"
        )

    with col2:
        rec_type = st.selectbox(
            "Type",
            ["Any", "Movie", "TV Show"],
            label_visibility="collapsed"
        )

    # ── Suggestion chips ──────────────────────────────────────────────────────
    st.markdown("<div style='font-size:0.8rem;color:#555;margin-bottom:10px'>Try these:</div>", unsafe_allow_html=True)
    suggestions = ["Squid Game", "Breaking Bad", "The Crown", "Bird Box", "Narcos"]
    cols = st.columns(5)
    for i, s in enumerate(suggestions):
        if cols[i].button(s, key=f"rec_{i}"):
            st.session_state['rec_title'] = s

    if 'rec_title' in st.session_state:
        liked_title = st.session_state['rec_title']

    # ── Get recommendations ───────────────────────────────────────────────────
    if st.button("🍿 Get Recommendations", type="primary") and liked_title:
        with st.spinner(f"Finding titles similar to '{liked_title}'..."):

            # Build smart query
            type_filter = f" Focus only on {rec_type}s." if rec_type != "Any" else ""
            query = (
                f"I really enjoyed watching '{liked_title}'. "
                f"What are similar Netflix titles I would like based on genre, "
                f"theme, tone and storytelling style?{type_filter}"
            )

            # Get RAG answer
            answer, sources = ask(query)

        # ── AI Answer ─────────────────────────────────────────────────────────
        st.markdown(f"""
            <div style='background:linear-gradient(135deg,#1a0000,#111);
                        border:1px solid rgba(229,9,20,0.3);
                        border-left:3px solid #E50914;
                        border-radius:12px;padding:20px;margin:16px 0'>
                <div style='font-size:0.7rem;color:#E50914;font-weight:600;
                            text-transform:uppercase;letter-spacing:0.1em;margin-bottom:10px'>
                    🤖 AI Recommendations
                </div>
                <div style='color:#ddd;font-size:0.95rem;line-height:1.7'>{answer}</div>
            </div>
        """, unsafe_allow_html=True)

        # ── Matched titles from dataset ────────────────────────────────────────
        st.markdown("<div style='font-size:0.9rem;font-weight:600;color:#fff;margin:20px 0 12px'>📋 Matched from Netflix catalogue:</div>", unsafe_allow_html=True)

        # Extract titles mentioned in sources
        matched_cards = ""
        shown = 0
        for source in sources:
            # Find matching title in dataframe
            for _, row in df.iterrows():
                if (str(row['title']).lower() in source.lower() and
                    str(row['title']).lower() != liked_title.lower() and
                    shown < 6):

                    content_icon = "🎥" if row['type'] == "Movie" else "📺"
                    duration = str(row['duration']) if pd.notna(row['duration']) else "N/A"
                    country = str(row['country']).split(',')[0].strip() if pd.notna(row['country']) else "Unknown"
                    description = str(row['description'])[:120] + "..." if pd.notna(row['description']) else ""
                    year = int(row['release_year']) if pd.notna(row['release_year']) else "N/A"
                    rating = str(row['rating']) if pd.notna(row['rating']) else "N/A"
                    genres = str(row['listed_in'])[:40] if pd.notna(row['listed_in']) else ""

                    matched_cards += f"""
                        <div style='background:#111;border:1px solid #222;border-radius:12px;
                                    padding:16px;margin-bottom:10px'>
                            <div style='display:flex;justify-content:space-between'>
                                <div style='font-size:1rem;font-weight:600;color:#fff'>
                                    {content_icon} {row['title']}
                                </div>
                                <span style='background:rgba(229,9,20,0.15);
                                             border:1px solid rgba(229,9,20,0.3);
                                             color:#E50914;padding:3px 10px;
                                             border-radius:20px;font-size:0.7rem;
                                             font-weight:600'>{rating}</span>
                            </div>
                            <div style='font-size:0.75rem;color:#555;margin-top:4px'>
                                📅 {year} &nbsp;|&nbsp; ⏱ {duration} &nbsp;|&nbsp; 🌍 {country}
                            </div>
                            <div style='font-size:0.78rem;color:#444;margin-top:8px;line-height:1.5'>
                                {description}
                            </div>
                            <div style='font-size:0.7rem;color:#333;margin-top:6px'>
                                🎭 {genres}
                            </div>
                        </div>
                    """
                    shown += 1
                    break

        if matched_cards:
            import streamlit.components.v1 as components
            components.html(f"""
                <style>
                    body {{ margin:0;background:transparent;font-family:Inter,sans-serif; }}
                    ::-webkit-scrollbar {{ width:4px; }}
                    ::-webkit-scrollbar-track {{ background:#1a1a1a; }}
                    ::-webkit-scrollbar-thumb {{ background:#E50914;border-radius:4px; }}
                </style>
                <div style='max-height:500px;overflow-y:auto;padding-right:8px'>
                    {matched_cards}
                </div>
            """, height=520, scrolling=False)
        else:
            st.markdown("""
                <div style='color:#444;font-size:0.85rem;margin-top:8px'>
                    💡 Try searching for a more popular title for better matches!
                </div>
            """, unsafe_allow_html=True)

        # ── Sources expander ──────────────────────────────────────────────────
        with st.expander("📄 Data sources used"):
            for s in sources:
                st.markdown(f"<div style='font-size:0.8rem;color:#555;margin-bottom:6px'>📊 {s[:100]}...</div>", unsafe_allow_html=True)

# ── Tab 7: Ask AI
with tab7:
    st.markdown("""
        <div style='margin-bottom:24px'>
            <div style='font-size:1.3rem;font-weight:600;color:#fff;margin-bottom:6px'>
                🤖 Ask the Netflix AI Analyst
            </div>
            <div style='font-size:0.85rem;color:#555'>
                Natural language questions answered from real Netflix data via RAG
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Init chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Suggestion chips
    suggestions = [
        "Which country produces the most content?",
        "What is the most popular genre?",
        "How has Netflix grown over the years?",
        "What rating has the most titles?",
        "When did Netflix add the most content?",
        "Compare movies vs TV shows",
    ]

    st.markdown("<div style='font-size:0.8rem;color:#555;margin-bottom:10px;text-transform:uppercase;letter-spacing:0.08em'>Quick questions</div>", unsafe_allow_html=True)
    auto_ask = None
    cols = st.columns(3)
    for i, s in enumerate(suggestions):
        if cols[i % 3].button(s, key=f"chip_{i}"):
            st.session_state['chat_input'] = s
            auto_ask = s

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # Chat history
    for msg in st.session_state.messages:
        if msg['role'] == 'user':
            st.markdown(f'<div class="chat-message-user">🧑 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message-ai"><div class="ai-label">🤖 Netflix AI</div>{msg["content"]}</div>', unsafe_allow_html=True)

    # Input
    question = st.text_input(
        "Ask anything:",
        placeholder="e.g. What genre dominates Netflix?",
        label_visibility="collapsed",
        key="chat_input"
    )

    col_btn, col_clear = st.columns([1, 5])
    ask_clicked = col_btn.button("Ask →", type="primary")
    if col_clear.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

    final_question = auto_ask or (question if ask_clicked else None)
    if final_question:
        st.session_state.messages.append({"role": "user", "content": final_question})
        with st.spinner("Analysing..."):
            answer, sources = ask(final_question)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        if auto_ask:
            st.session_state['chat_input'] = ''

        with st.expander("📄 Data sources used"):
            for s in sources:
                st.markdown(f'<span class="source-pill">📊 {s[:80]}...</span>', unsafe_allow_html=True)

        st.rerun()