import streamlit as st
import pandas as pd
import requests
from difflib import get_close_matches
from reinforcement import ReinforcementAgent
from utils import canonical_schema, simple_clean_value, compare_tables

st.set_page_config(layout="wide", page_title="Schema Mapper & Reinforcement FE")

st.title("Schema Mapper & Reinforcement — Frontend")
st.markdown("Upload CSV or choose a sample. This app runs in **mock** mode if backend endpoints are not available. "
            "It demonstrates FE, manual mapping UI, suggestive mapping, before/after tabular view, and a small reinforcement learner that remembers accepted fixes.")

# Agent (simple persistence in local file)
agent = ReinforcementAgent(stats_path='rl_stats.json')

# Sidebar: choose mode
st.sidebar.header("Mode & Backend")
use_backend = st.sidebar.checkbox("Use real backend endpoints", value=False)
backend_url = st.sidebar.text_input("Backend base URL (if using real backend)", value="http://localhost:8000")

# File upload / sample
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])


if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview — Raw Input (first 10 rows)")
    st.dataframe(df.head(10))

    # Simple suggestive mapping: try to map input columns to canonical schema using fuzzy matching
    st.subheader("Suggestive Mapping (automatic)")
    can_schema = canonical_schema()
    cols = list(df.columns)
    mapping = {}
    confidence = {}
    for c in cols:
        if c in can_schema:
            mapping[c] = c
            confidence[c] = 0.99
        else:
            matches = get_close_matches(c, can_schema.keys(), n=1, cutoff=0.6)
            if matches:
                mapping[c] = matches[0]
                conf_mem = agent.get_confidence(f"{c}:::{matches[0]}")
                confidence[c] = min(0.95, 0.5 + conf_mem * 0.5)
            else:
                mapping[c] = None
                confidence[c] = 0.1

    # Display mapping suggestions
    st.write("Mappings suggested (column -> canonical) — click Accept to apply and teach the agent.")
    for c in cols:
        cols1, cols2, cols3 = st.columns([3,4,2])
        cols1.write(f"**{c}**")
        suggested = mapping[c] if mapping[c] is not None else "— no suggestion —"
        cols2.write(f"Suggested: **{suggested}**  \nConfidence: {confidence[c]:.2f}")
        if cols3.button("Accept", key=f"accept::{c}"):
            if mapping[c]:
                agent.update(f"{c}:::{mapping[c]}", reward=1)
                st.success(f"Accepted mapping {c} → {mapping[c]} (agent updated)")
        if cols3.button("Reject", key=f"reject::{c}"):
            if mapping[c]:
                agent.update(f"{c}:::{mapping[c]}", reward=0)
                st.info(f"Rejected mapping {c} → {mapping[c]} (agent updated)")

    # Manual mapping
    st.subheader("Manual Mapping Editor")
    manual_map = {}
    for c in cols:
        manual_map[c] = st.selectbox(f"Map column '{c}' to canonical", options=["--none--"] + list(can_schema.keys()), index=0, key=f"mm_{c}")

    if st.button("Apply Manual Mapping"):
        applied = {c: v for c, v in manual_map.items() if v != "--none--"}
        st.success(f"Applied manual mapping for {len(applied)} columns.")
        for c, v in applied.items():
            agent.update(f"{c}:::{v}", reward=1)

    # Before / After comparison
    st.subheader("Before / After — Tabular Comparison (first 20 rows)")
    clean_df = df.copy()
    rename_map = {c: manual_map[c] for c in cols if manual_map[c] != "--none--"}
    for c in cols:
        if manual_map[c] == "--none--" and mapping[c] is not None:
            rename_map[c] = mapping[c]
    clean_df = clean_df.rename(columns=rename_map)
    for col in clean_df.columns:
        clean_df[col] = clean_df[col].apply(simple_clean_value)

    before_tab, after_tab = st.columns(2)
    with before_tab:
        st.write("Before (raw)")
        st.dataframe(df.head(20))
    with after_tab:
        st.write("After (cleaned & mapped)")
        st.dataframe(clean_df.head(20))

    # Download
    st.subheader("Download")
    from io import StringIO
    csv_buf = StringIO()
    clean_df.to_csv(csv_buf, index=False)
    b = csv_buf.getvalue().encode('utf-8')
    st.download_button("Download cleaned CSV", data=b, file_name="cleaned_dataset.csv", mime="text/csv")

    st.sidebar.markdown("---")
    st.sidebar.write("RL stats snapshot:")
    st.sidebar.json(agent.get_all_stats())
else:
    st.info("Please upload a CSV file to proceed.")
    st.stop()


