"""
ModelLens AI -- entry point.

Run with:
    streamlit run app.py
"""

import streamlit as st

from src.database.database import init_db

st.set_page_config(page_title="ModelLens AI", page_icon="🔍", layout="wide")

init_db()

dashboard = st.Page("pages/dashboard.py", title="Dashboard", icon="📊", default=True)
new_analysis = st.Page("pages/analysis.py", title="New Analysis", icon="🧪")
history = st.Page("pages/history.py", title="History", icon="🕘")

pg = st.navigation([dashboard, new_analysis, history])

st.sidebar.title("🔍 ModelLens AI")
st.sidebar.caption("ML Model Failure Analysis Platform")

pg.run()
