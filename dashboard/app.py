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