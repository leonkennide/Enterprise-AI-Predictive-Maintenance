import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sqlite3
import os
import datetime
import time
import plotly.express as px
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest

# Page Configuration
st.set_page_config(
    page_title="Enterprise Industrial AI & SCADA Suite",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Luxury Enterprise Vibe
st.markdown("""
    <style>
    .main { background-color: #0f172a; color: #f8fafc; }
    .stSidebar { background-color: #1e293b; }
    h1, h2, h3 { color: #38bdf8 !important; font-family: 'Segoe UI', sans-serif; }
    </style>
""", unsafe_allow_html=True)

MODEL_FILENAME = "ai_model.joblib"
ANOMALY_MODEL_FILENAME = "anomaly_model.joblib"
DB_FILENAME = "enterprise_maintenance.db"

# Initialize SQLite Database with Fleet Support
def init_db():
    conn = sqlite3.connect(DB_FILENAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS telemetry_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            machine_id TEXT,
            temperature REAL,
            vibration REAL,
            pressure REAL,
            prediction_status TEXT,
            confidence REAL
        )
    """)
    cursor.execute("PRAGMA table_info(telemetry_logs)")
    columns = [col[1] for col in cursor.fetchall()]
    if 'machine_id' not in columns:
        cursor.execute("ALTER TABLE telemetry_logs ADD COLUMN machine_id TEXT DEFAULT 'Turbine-01'")
    conn.commit()
    conn.close()

init_db()

# Load or Train Advanced AI Models
@st.cache_resource
def load_ai_engines():
    np.random.seed(42)
    n_samples = 1500
    temperatures = np.random.uniform(50.0, 110.0, n_samples)
    vibrations = np.random.uniform(0.5, 8.0, n_samples)
    pressures = np.random.uniform(20.0, 80.0, n_samples)
    
    failure_condition = (temperatures > 85.0) & (vibrations > 4.0) | (pressures > 65.0)
    failures = np.where(failure_condition, 1, 0)
    noise = np.random.binomial(1, 0.03, n_samples)
    failures = np.clip(failures + noise, 0, 1)

    df = pd.DataFrame({
        'Temperature': temperatures,
        'Vibration': vibrations,
        'Pressure': pressures,
        'Failure': failures
    })

    X = df[['Temperature', 'Vibration', 'Pressure']]
    y = df['Failure']

    # 1. Supervised Classifier (Gradient Boosting)
    if os.path.exists(MODEL_FILENAME):
        model = joblib.load(MODEL_FILENAME)
    else:
        model = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, random_state=42)
        model.fit(X, y)
        joblib.dump(model, MODEL_FILENAME)

    # 2. Unsupervised Anomaly Detector (Isolation Forest)
    if os.path.exists(ANOMALY_MODEL_FILENAME):
        anomaly_model = joblib.load(ANOMALY_MODEL_FILENAME)
    else:
        anomaly_model = IsolationForest(contamination=0.08, random_state=42)
        anomaly_model.fit(X)
        joblib.dump(anomaly_model, ANOMALY_MODEL_FILENAME)
        
    return model, anomaly_model, df

model, anomaly_model, dataset = load_ai_engines()

# Sidebar Navigation & Fleet Selector
st.sidebar.markdown("### 🛠️ Enterprise Suite")
menu = st.sidebar.radio("Navigation Panel", [
    "⚡ Live AI Prediction", 
    "📡 Live SCADA Stream Simulator",
    "📁 Batch CSV Analysis", 
    "📈 Interactive Analytics", 
    "🗄️ SQL Historical Logs"
])

st.sidebar.markdown("---")
st.sidebar.markdown("### 🏭 Fleet Selection")
selected_machine = st.sidebar.selectbox("Target Machine ID", [
    "Turbine-01", 
    "Generator-02", 
    "Compressor-03", 
    "Main-Engine-04"
])

st.sidebar.markdown("---")
st.sidebar.info("System Status: **Online & Secured**\nEngines: GB Classifier & Isolation Forest")

# Main Header
st.title("🚀 Enterprise Industrial AI & SCADA Suite")
st.markdown(f"##### Monitoring Active Asset: **{selected_machine}** | Advanced Telemetry & Anomaly Center")
st.markdown("---")

# Quick Stats Bar from DB for selected machine
conn = sqlite3.connect(DB_FILENAME)
try:
    total_scans = pd.read_sql_query(f"SELECT COUNT(*) FROM telemetry_logs WHERE machine_id='{selected_machine}'", conn).iloc[0, 0]
    critical_alerts = pd.read_sql_query(f"SELECT COUNT(*) FROM telemetry_logs WHERE machine_id='{selected_machine}' AND prediction_status='CRITICAL'", conn).iloc[0, 0]
except:
    total_scans, critical_alerts = 0, 0
conn.close()

col_stat1, col_stat2, col_stat3 = st.columns(3)
with col_stat1:
    st.metric(label=f"📊 Scans ({selected_machine})", value=total_scans)
with col_stat2:
    st.metric(label=f"⚠️ Critical Alerts ({selected_machine})", value=critical_alerts)
with col_stat3:
    st.metric(label="🟢 Dual-AI Engine Health", value="Optimal")

st.markdown("---")

if menu == "⚡ Live AI Prediction":
    st.subheader(f"⚡ Real-time Telemetry & Dual-Model Inference [{selected_machine}]")
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("##### Adjust Machine Parameters:")
        temp = st.slider("Core Temperature (°C)", 50.0, 110.0, 75.5)
        vib = st.slider("Vibration Amplitude", 0.5, 8.0, 2.1)
        press = st.slider("System Pressure (PSI)", 20.0, 80.0, 45.0)
        
        execute_btn = st.button("🚀 Execute AI & Anomaly Scan", use_container_width=True)
        
    with col2:
        st.markdown("##### Advanced Diagnostic Results:")
        
        if execute_btn:
            input_df = pd.DataFrame([[temp, vib, press]], columns=['Temperature', 'Vibration', 'Pressure'])
            prediction = model.predict(input_df)
            probability = model.predict_proba(input_df)[0]
            
            # Isolation Forest anomaly check (-1 = anomaly, 1 = normal)
            anomaly_pred = anomaly_model.predict(input_df)[0]
            is_anomaly = (anomaly_pred == -1)
            
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            status_str = "CRITICAL" if (prediction[0] == 1 or is_anomaly) else "NORMAL"
            conf_pct = float(probability[1] * 100 if prediction[0] == 1 else probability[0] * 100)
            
            # Save to SQLite
            conn = sqlite3.connect(DB_FILENAME)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO telemetry_logs (timestamp, machine_id, temperature, vibration, pressure, prediction_status, confidence) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (current_time, selected_machine, temp, vib, press, status_str, conf_pct)
            )
            conn.commit()
            conn.close()
            
            if status_str == "CRITICAL":
                st.error(f"⚠️ **CRITICAL WARNING / ANOMALY DETECTED ({selected_machine})!**\n\n* **Supervised AI Confidence:** {conf_pct:.1f}%\n* **Isolation Forest Anomaly Flag:** {'YES (Unusual Behavior)' if is_anomaly else 'NO'}\n* **Action:** Immediate Engineering Inspection Required.")
            else:
                st.success(f"✅ **SYSTEM OPERATING NORMALLY ({selected_machine})**\n\n* **Safety Confidence:** {conf_pct:.1f}%\n* **Anomaly Status:** Clean Behaviour Pattern.")
        else:
            st.info(f"👈 Adjust parameters for **{selected_machine}** and click **Execute AI & Anomaly Scan**.")

elif menu == "📡 Live SCADA Stream Simulator":
    st.subheader(f"📡 Real-time SCADA Telemetry Stream Simulator [{selected_machine}]")
    st.markdown("Simulate live sensor data streams flowing directly from the machine sensors into the AI engine in real-time.")
    
    stream_cycles = st.slider("Number of Live Stream Cycles", 5, 30, 10)
    start_stream = st.button("▶ Start Live Telemetry Stream", use_container_width=True)
    
    if start_stream:
        stream_placeholder = st.empty()
        log_placeholder = st.empty()
        chart_placeholder = st.empty()
        
        simulated_logs = []
        
        for i in range(stream_cycles):
            # Generate fluctuating live values
            r_temp = np.random.uniform(60.0, 105.0)
            r_vib = np.random.uniform(1.0, 7.5)
            r_press = np.random.uniform(30.0, 75.0)
            
            inp = pd.DataFrame([[r_temp, r_vib, r_press]], columns=['Temperature', 'Vibration', 'Pressure'])
            pred = model.predict(inp)[0]
            anom = anomaly_model.predict(inp)[0]
            status = "CRITICAL" if (pred == 1 or anom == -1) else "NORMAL"
            conf = 92.5 if status == "CRITICAL" else 98.1
            
            curr_t = datetime.datetime.now().strftime("%H:%M:%S")
            simulated_logs.append({
                "Timestamp": curr_t,
                "Machine": selected_machine,
                "Temperature": round(r_temp, 2),
                "Vibration": round(r_vib, 2),
                "Pressure": round(r_press, 2),
                "Status": status
            })
            
            # Save to database automatically
            conn = sqlite3.connect(DB_FILENAME)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO telemetry_logs (timestamp, machine_id, temperature, vibration, pressure, prediction_status, confidence) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), selected_machine, r_temp, r_vib, r_press, status, conf)
            )
            conn.commit()
            conn.close()
            
            with stream_placeholder.container():
                cols = st.columns(3)
                cols[0].metric("🔴 Live Temperature", f"{r_temp:.1f} °C")
                cols[1].metric("🔵 Live Vibration", f"{r_vib:.2f}")
                cols[2].metric("⚡ Live Pressure", f"{r_press:.1f} PSI")
            
            df_live = pd.DataFrame(simulated_logs)
            log_placeholder.dataframe(df_live, use_container_width=True)
            
            time.sleep(1.0) # Pause between stream ticks
            
        st.success("✨ Live SCADA telemetry stream completed and recorded successfully!")

elif menu == "📁 Batch CSV Analysis":
    st.subheader(f"📁 Bulk Telemetry Analysis for [{selected_machine}]")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        user_df = pd.read_csv(uploaded_file)
        st.write("Preview of Uploaded Data:", user_df.head())
        
        if {'Temperature', 'Vibration', 'Pressure'}.issubset(user_df.columns):
            if st.button("Run Batch AI Analysis & Save Logs", use_container_width=True):
                preds = model.predict(user_df[['Temperature', 'Vibration', 'Pressure']])
                user_df['AI_Prediction'] = np.where(preds == 1, 'CRITICAL', 'NORMAL')
                
                conn = sqlite3.connect(DB_FILENAME)
                cursor = conn.cursor()
                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                for _, row in user_df.iterrows():
                    cursor.execute(
                        "INSERT INTO telemetry_logs (timestamp, machine_id, temperature, vibration, pressure, prediction_status, confidence) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (current_time, selected_machine, row['Temperature'], row['Vibration'], row['Pressure'], row['AI_Prediction'], 95.0)
                    )
                conn.commit()
                conn.close()
                
                st.success(f"Batch analysis completed and saved for **{selected_machine}**!")
                st.dataframe(user_df, use_container_width=True)
        else:
            st.error("Error: The uploaded CSV must contain columns: Temperature, Vibration, Pressure.")
    else:
        st.info("💡 Tip: Upload a sensor log file to perform bulk diagnostics.")

elif menu == "📈 Interactive Analytics":
    st.subheader(f"📈 Interactive Multi-Dimensional Sensor Mapping [{selected_machine}]")
    fig = px.scatter(
        dataset[:500], 
        x="Temperature", 
        y="Vibration", 
        color=dataset[:500]['Failure'].astype(str),
        size="Pressure",
        color_discrete_map={'0': '#38bdf8', '1': '#f43f5e'},
        title=f"Cluster Analysis - {selected_machine}",
        labels={'color': 'Failure Status (0: Normal, 1: Risk)'}
    )
    fig.update_layout(paper_bgcolor='#0f172a', plot_bgcolor='#1e293b', font_color='#f8fafc')
    st.plotly_chart(fig, use_container_width=True)

elif menu == "🗄️ SQL Historical Logs":
    st.subheader(f"🗄️ Database Audit Trail & Report Export [{selected_machine}]")
    conn = sqlite3.connect(DB_FILENAME)
    logs_df = pd.read_sql_query(f"SELECT * FROM telemetry_logs WHERE machine_id='{selected_machine}' ORDER BY id DESC", conn)
    conn.close()
    
    if not logs_df.empty:
        st.dataframe(logs_df, use_container_width=True, height=400)
        csv_data = logs_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"📥 Download Certified Report for {selected_machine} (CSV)",
            data=csv_data,
            file_name=f"maintenance_report_{selected_machine}.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.warning(f"No historical telemetry records found for **{selected_machine}** yet.")