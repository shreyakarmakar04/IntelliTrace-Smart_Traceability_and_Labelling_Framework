import streamlit as st  ### for final showing the results using streamlit -> UI
import sqlite3
import pandas as pd
import os
import time
import psutil  # pip install psutil

# === CONFIG ===
DB_PATH = r'C:\Users\HARSHITHA\Downloads\Python\inspection_results.db'
TABLE_NAME = 'FinalInspectionResults'

# === Streamlit UI Setup ===
st.set_page_config(page_title="📦 Batch Inspection Viewer", layout="centered")
st.title("🔍 Smart Product Inspection Lookup")

# === User Input ===
batch_id = st.text_input("Enter Batch ID to search:")

# === Database Query Function ===
def fetch_batch_data(batch_id):
    try:
        conn = sqlite3.connect(DB_PATH)
        # ✅ Correct syntax for column with space
        query = f"SELECT * FROM {TABLE_NAME} WHERE [Batch ID] = ?"
        df = pd.read_sql_query(query, conn, params=(batch_id,))
        conn.close()
        return df
    except Exception as e:
        st.error(f"Database error: {e}")
        return pd.DataFrame()

# === When User Submits a Batch ID ===
if batch_id:
    result_df = fetch_batch_data(batch_id)

    if result_df.empty:
        st.error(f"❌ No data found for Batch ID: {batch_id}")
    else:
        st.success(f"✅ Found {len(result_df)} record(s) for Batch ID: {batch_id}")
        st.dataframe(result_df)

# === Divider & Exit Button ===
st.markdown("---")
if st.button("❌ Exit App"):
    st.warning("App is shutting down... Please close this browser tab.")
    st.markdown("## ✅ Streamlit server will now stop.")
    time.sleep(2)

    # === Kill Streamlit process ===
    import subprocess
    current_pid = os.getpid()
    parent = psutil.Process(current_pid)
    for child in parent.children(recursive=True):
        child.kill()
    parent.kill()