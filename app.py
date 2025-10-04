import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import time
from collections import deque
from sklearn.ensemble import IsolationForest

# --- Page Configuration ---
st.set_page_config(
    page_title="Real-Time Anomaly Detection",
    layout="wide"
)

# --- Session State Initialization ---
if "running" not in st.session_state:
    st.session_state.running = False
if "data" not in st.session_state:
    st.session_state.data = deque(maxlen=100) # Deque for calculation window
if "anomaly_log" not in st.session_state:
    st.session_state.anomaly_log = []
if 'price' not in st.session_state:
    st.session_state.price = 100.0
# Add state to store all data and anomalies for persistent plotting
if 'all_data' not in st.session_state:
    st.session_state.all_data = []
if 'all_anomalies' not in st.session_state:
    st.session_state.all_anomalies = []
if 'time_step' not in st.session_state:
    st.session_state.time_step = 0

# --- Anomaly Detection Algorithms ---
def detect_anomalies_zscore(series, threshold):
    """Detect anomalies using the Z-Score method."""
    if len(series) < 2:
        return np.array([False] * len(series))
    mean = np.mean(series)
    std = np.std(series)
    z_scores = (series - mean) / (std + 1e-9)
    return np.abs(z_scores) > threshold

def detect_anomalies_mad(series, threshold):
    """Detect anomalies using Median Absolute Deviation (MAD)."""
    if len(series) < 2:
        return np.array([False] * len(series))
    median = np.median(series)
    mad = np.median(np.abs(series - median))
    modified_z_scores = 0.6745 * (series - median) / (mad + 1e-9)
    return np.abs(modified_z_scores) > threshold

def detect_anomalies_isolation_forest(series, contamination):
    """Detect anomalies using Isolation Forest."""
    if len(series) < 2:
        return np.array([False] * len(series))
    series_reshaped = series.reshape(-1, 1)
    model = IsolationForest(contamination=contamination, random_state=42)
    preds = model.fit_predict(series_reshaped)
    return preds == -1

# --- Sidebar Controls ---
st.sidebar.title("⚙️ Controls")

# Simulation control buttons
if st.sidebar.button("▶️ Start"):
    st.session_state.running = True
if st.sidebar.button("⏹️ Stop"):
    st.session_state.running = False
if st.sidebar.button("🔄 Reset"):
    st.session_state.running = False
    st.session_state.data.clear()
    st.session_state.anomaly_log.clear()
    st.session_state.price = 100.0
    st.session_state.all_data.clear()
    st.session_state.all_anomalies.clear()
    st.session_state.time_step = 0

st.sidebar.markdown("---")

# Algorithm selection and parameter sliders
method = st.sidebar.selectbox(
    "Anomaly Detection Algorithm",
    ["Z-Score", "Median Absolute Deviation (MAD)", "Isolation Forest"]
)

window_size = st.sidebar.slider("Rolling Window Size", 10, 500, 100)

# Recreate the deque with the new maxlen if the window size changes
if window_size != st.session_state.data.maxlen:
    st.session_state.data = deque(list(st.session_state.data), maxlen=window_size)

# Conditionally display parameters based on the selected method
if method == "Z-Score":
    threshold = st.sidebar.slider("Z-Score Threshold", 1.0, 5.0, 3.0, step=0.1, key="z_thresh")
elif method == "Median Absolute Deviation (MAD)":
    threshold = st.sidebar.slider("MAD Threshold", 1.0, 10.0, 3.5, step=0.1, key="mad_thresh")
else:
    contamination = st.sidebar.slider("Contamination", 0.01, 0.5, 0.05, step=0.01, key="iso_contam")

# --- Main Application Layout ---
st.title("📈 Real-Time Anomaly Detection Dashboard")

price_chart_placeholder = st.empty()

st.sidebar.markdown("---")
st.sidebar.title("📝 Anomaly Log")
log_placeholder = st.sidebar.empty()


def draw_dashboard():
    """Draws the main chart and the anomaly log."""
    # --- Charting ---
    fig_price = go.Figure()

    # Trace for all historical data
    time_indices = list(range(st.session_state.time_step))
    fig_price.add_trace(go.Scatter(
        x=time_indices, 
        y=st.session_state.all_data, 
        mode='lines', 
        name='Value', 
        line=dict(color='#3b82f6', width=2)
    ))

    # Trace for all historical anomalies
    if st.session_state.all_anomalies:
        anomaly_x, anomaly_y = zip(*st.session_state.all_anomalies)
        fig_price.add_trace(go.Scatter(
            x=list(anomaly_x), 
            y=list(anomaly_y),
            mode='markers', 
            name='Anomaly',
            marker=dict(color='#ef4444', size=12, symbol='x', line=dict(width=2))
        ))

    fig_price.update_layout(
        title="Live Value & Anomaly Detection", 
        xaxis_title="Time Step", 
        yaxis_title="Value",
        hovermode='x unified',
        template='plotly_dark'
    )
    price_chart_placeholder.plotly_chart(fig_price, use_container_width=True)

    # --- Log Display ---
    if st.session_state.anomaly_log:
        log_text = "\n\n".join([f"🔴 {log}" for log in st.session_state.anomaly_log])
        log_placeholder.markdown(f"```\n{log_text}\n```")
    else:
        log_placeholder.info("No anomalies detected yet.")

# --- Main Simulation Loop ---
while st.session_state.running:
    # --- Data Generation ---
    drift = np.random.normal(0, 0.2)
    shock = np.random.normal(0, 10) if np.random.rand() < 0.05 else 0
    st.session_state.price += drift + shock
    st.session_state.price = max(st.session_state.price, 10)
    
    # Append to both the calculation deque and the persistent data list
    st.session_state.data.append(st.session_state.price)
    st.session_state.all_data.append(st.session_state.price)
    
    st.session_state.time_step += 1
    
    series = np.array(st.session_state.data, dtype=float)

    # --- Anomaly Detection ---
    if len(series) >= window_size:
        if method == "Z-Score":
            anomalies = detect_anomalies_zscore(series, threshold)
        elif method == "Median Absolute Deviation (MAD)":
            anomalies = detect_anomalies_mad(series, threshold)
        else:
            anomalies = detect_anomalies_isolation_forest(series, contamination)
        
        # Log the newest anomaly if detected
        if anomalies[-1]:
            log_entry = f"[{time.strftime('%H:%M:%S')}] Anomaly: {series[-1]:.2f}"
            st.session_state.anomaly_log.insert(0, log_entry)
            st.session_state.all_anomalies.append((st.session_state.time_step - 1, series[-1]))

            if len(st.session_state.anomaly_log) > 50:
                st.session_state.anomaly_log = st.session_state.anomaly_log[:50]
    
    # Redraw the dashboard with new data
    draw_dashboard()
    
    time.sleep(0.1)


# --- Static Display when Paused or Stopped ---
# This ensures the chart and log remain visible after stopping the simulation.
if not st.session_state.running:
    draw_dashboard()

    # --- Status Message ---
    if st.session_state.time_step == 0:
        st.info("👈 Press **Start** in the sidebar to begin the simulation.")
    else:
        st.info("Simulation paused. Press 'Start' to resume or 'Reset' to clear.")

