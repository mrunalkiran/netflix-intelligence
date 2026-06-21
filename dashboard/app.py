import streamlit as st
import pandas as pd
import plotly.express as px
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from pipeline.analyse import (
    load_clean, content_type_split, top_genres,
    content_per_year, top_countries, ratings_distribution
)
from rag.query import ask

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Netflix Intelligence",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Netflix Content Intelligence Platform")
st.caption("Analytics dashboard + AI analyst powered by RAG")

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def get_data():
    return load_clean()

df = get_data()

# ── Sidebar filters ───────────────────────────────────────────────────────────
st.sidebar.header("Filters")
content_type = st.sidebar.multiselect(
    "Content Type",
    options=df['type'].unique(),
    default=df['type'].unique()
)
df = df[df['type'].isin(content_type)]

# ── Metrics row ───────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Titles", f"{len(df):,}")
col2.metric("Movies", f"{len(df[df['type']=='Movie']):,}")
col3.metric("TV Shows", f"{len(df[df['type']=='TV Show']):,}")
col4.metric("Countries", f"{df['country'].nunique():,}")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🎭 Genres", "🌍 Countries", "📅 Growth", "🤖 Ask AI"
])

# ── Tab 1: Genres ─────────────────────────────────────────────────────────────
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top Genres")
        genres = top_genres(df)
        fig = px.bar(
            genres, x='count', y='listed_in',
            orientation='h',
            color='count',
            color_continuous_scale='Reds',
            labels={'listed_in': 'Genre', 'count': 'Titles'}
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Movies vs TV Shows")
        type_split = content_type_split(df)
        fig2 = px.pie(
            type_split, values='count', names='type',
            color_discrete_sequence=['#E50914', '#221F1F'],
            hole=0.4
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Ratings Distribution")
    ratings = ratings_distribution(df)
    # Filter out bad rows
    ratings = ratings[ratings['rating'].str.len() < 10]
    fig3 = px.bar(
        ratings, x='rating', y='count',
        color='count', color_continuous_scale='Reds',
        labels={'rating': 'Rating', 'count': 'Titles'}
    )
    st.plotly_chart(fig3, use_container_width=True)

# ── Tab 2: Countries ──────────────────────────────────────────────────────────
with tab2:
    st.subheader("Top Content Producing Countries")
    countries = top_countries(df)
    fig = px.bar(
        countries, x='country', y='count',
        color='count', color_continuous_scale='Reds',
        labels={'country': 'Country', 'count': 'Titles'}
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("World Map")
    fig2 = px.choropleth(
        countries, locations='country',
        locationmode='country names',
        color='count',
        color_continuous_scale='Reds',
        title='Netflix Content by Country'
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── Tab 3: Growth ─────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Netflix Content Growth Over Years")
    per_year = content_per_year(df)
    per_year = per_year[per_year['year_added'] >= 2010]
    fig = px.area(
        per_year, x='year_added', y='count',
        color_discrete_sequence=['#E50914'],
        labels={'year_added': 'Year', 'count': 'Titles Added'},
        markers=True
    )
    st.plotly_chart(fig, use_container_width=True)

    # Peak year insight
    peak = per_year.loc[per_year['count'].idxmax()]
    st.info(f"📈 Peak year: **{int(peak['year_added'])}** with **{int(peak['count'])}** titles added")

# ── Tab 4: Ask AI ─────────────────────────────────────────────────────────────
with tab4:
    st.subheader("🤖 Ask the Netflix AI Analyst")
    st.caption("Ask anything about Netflix content — answers grounded in real data")

    # Suggested questions
    suggestions = [
        "Which country produces the most content?",
        "What is the most popular genre?",
        "How has Netflix grown over the years?",
        "What rating has the most titles?",
        "When did Netflix add the most content?",
    ]
    st.markdown("**Try asking:**")
    cols = st.columns(3)
    for i, s in enumerate(suggestions):
        if cols[i % 3].button(s, key=f"s_{i}"):
            st.session_state['question'] = s

    question = st.text_input(
        "Your question:",
        value=st.session_state.get('question', ''),
        placeholder="e.g. What genre dominates Netflix?"
    )

    if st.button("Ask", type="primary") and question:
        with st.spinner("Analysing Netflix data..."):
            answer, sources = ask(question)

        st.success(answer)

        with st.expander("📄 Sources used"):
            for i, s in enumerate(sources, 1):
                st.markdown(f"**{i}.** {s}")