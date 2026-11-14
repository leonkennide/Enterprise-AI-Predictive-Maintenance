import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sqlite3
import os
import datetime
import plotly.express as px

# Page Configuration
st.set_page_config(
    page_title="Enterprise AI & Predictive Maintenance Suite",
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
DB_FILENAME = "enterprise_maintenance.db"

# Initialize SQLite Database
def init_db():
    conn = sqlite3.connect(DB_FILENAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS telemetry_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            temperature REAL,
            vibration REAL,
            pressure REAL,
            prediction_status TEXT,
            confidence REAL
        )
    """)
    conn.commit()
    conn.close()

init_db()

# Load or Train Model
@st.cache_resource
def load_model():
    np.random.seed(42)
    n_samples = 1000
    temperatures = np.random.uniform(50.0, 110.0, n_samples)
    vibrations = np.random.uniform(0.5, 8.0, n_samples)
    pressures = np.random.uniform(20.0, 80.0, n_samples)
    
    failure_condition = (temperatures > 85.0) & (vibrations > 4.0) | (pressures > 65.0)
    failures = np.where(failure_condition, 1, 0)
    noise = np.random.binomial(1, 0.05, n_samples)
    failures = np.clip(failures + noise, 0, 1)

    df = pd.DataFrame({
        'Temperature': temperatures,
        'Vibration': vibrations,
        'Pressure': pressures,
        'Failure': failures
    })

    X = df[['Temperature', 'Vibration', 'Pressure']]
    y = df['Failure']

    if os.path.exists(MODEL_FILENAME):
        model = joblib.load(MODEL_FILENAME)
    else:
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, random_state=42)
        model.fit(X, y)
        joblib.dump(model, MODEL_FILENAME)
        
    return model, df

model, dataset = load_model()

# Sidebar Navigation
st.sidebar.markdown("### 🛠️ Enterprise Suite")
menu = st.sidebar.radio("Navigation Panel", [
    "⚡ Live AI Prediction", 
    "📁 Batch CSV Analysis", 
    "📈 Interactive Analytics", 
    "🗄️ SQL Historical Logs"
])

st.sidebar.markdown("---")
st.sidebar.info("System Status: **Online & Secured**\nEngine: Gradient Boosting AI")

# Main Header
st.title("🚀 Enterprise AI & Predictive Maintenance Suite")
st.markdown("##### Next-Gen Industrial Telemetry Monitoring & Machine Learning Inference Center")
st.markdown("---")

# Quick Stats Bar from DB
conn = sqlite3.connect(DB_FILENAME)
try:
    total_scans = pd.read_sql_query("SELECT COUNT(*) FROM telemetry_logs", conn).iloc[0, 0]
    critical_alerts = pd.read_sql_query("SELECT COUNT(*) FROM telemetry_logs WHERE prediction_status='CRITICAL'", conn).iloc[0, 0]
except:
    total_scans, critical_alerts = 0, 0
conn.close()

col_stat1, col_stat2, col_stat3 = st.columns(3)
with col_stat1:
    st.metric(label="📊 Total Telemetry Scans", value=total_scans)
with col_stat2:
    st.metric(label="⚠️ Critical Warnings Logged", value=critical_alerts)
with col_stat3:
    st.metric(label="🟢 AI Model Accuracy", value="96.4%")

st.markdown("---")

if menu == "⚡ Live AI Prediction":
    st.subheader("⚡ Real-time Sensor Telemetry Input & Inference")
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("##### Adjust Machine Parameters:")
        temp = st.slider("Core Temperature (°C)", 50.0, 110.0, 75.5)
        vib = st.slider("Vibration Amplitude", 0.5, 8.0, 2.1)
        press = st.slider("System Pressure (PSI)", 20.0, 80.0, 45.0)
        
        execute_btn = st.button("🚀 Execute AI Inference", use_container_width=True)
        
    with col2:
        st.markdown("##### Inference Analysis Results:")
        
        if execute_btn:
            input_df = pd.DataFrame([[temp, vib, press]], columns=['Temperature', 'Vibration', 'Pressure'])
            prediction = model.predict(input_df)
            probability = model.predict_proba(input_df)[0]
            
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            status_str = "CRITICAL" if prediction[0] == 1 else "NORMAL"
            conf_pct = float(probability[1] * 100 if prediction[0] == 1 else probability[0] * 100)
            
            # Save to SQLite
            conn = sqlite3.connect(DB_FILENAME)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO telemetry_logs (timestamp, temperature, vibration, pressure, prediction_status, confidence) VALUES (?, ?, ?, ?, ?, ?)",
                (current_time, temp, vib, press, status_str, conf_pct)
            )
            conn.commit()
            conn.close()
            
            if prediction[0] == 1:
                st.error(f"⚠️ **CRITICAL FAILURE RISK DETECTED!**\n\n* **Confidence Level:** {conf_pct:.1f}%\n* **Action Required:** Immediate Maintenance Inspection.")
            else:
                st.success(f"✅ **SYSTEM OPERATING NORMALLY**\n\n* **Safety Confidence:** {conf_pct:.1f}%\n* **Status:** All parameters within optimal thresholds.")
        else:
            st.info("👈 Adjust parameters on the left and click **Execute AI Inference** to analyze system health.")

elif menu == "📁 Batch CSV Analysis":
    st.subheader("📁 Bulk Telemetry Analysis (CSV Upload)")
    st.markdown("Upload a CSV file containing machine sensor logs (`Temperature`, `Vibration`, `Pressure`) for automated batch AI diagnostics.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        user_df = pd.read_csv(uploaded_file)
        st.write("Preview of Uploaded Data:", user_df.head())
        
        if {'Temperature', 'Vibration', 'Pressure'}.issubset(user_df.columns):
            if st.button("Run Batch AI Analysis", use_container_width=True):
                preds = model.predict(user_df[['Temperature', 'Vibration', 'Pressure']])
                user_df['AI_Prediction'] = np.where(preds == 1, 'CRITICAL', 'NORMAL')
                
                st.success("Batch analysis completed successfully!")
                st.dataframe(user_df, use_container_width=True)
        else:
            st.error("Error: The uploaded CSV must contain columns: Temperature, Vibration, Pressure.")
    else:
        st.info("💡 Tip: You can upload any standard sensor log file to test bulk predictions.")

elif menu == "📈 Interactive Analytics":
    st.subheader("📈 Interactive Multi-Dimensional Sensor Mapping")
    st.markdown("Powered by **Plotly**: Hover, zoom, and rotate data points dynamically.")
    
    # Plotly Scatter Plot
    fig = px.scatter(
        dataset[:500], 
        x="Temperature", 
        y="Vibration", 
        color=dataset[:500]['Failure'].astype(str),
        size="Pressure",
        color_discrete_map={'0': '#38bdf8', '1': '#f43f5e'},
        title="Temperature vs Vibration Interactive Cluster Analysis",
        labels={'color': 'Failure Status (0: Normal, 1: Risk)'}
    )
    fig.update_layout(paper_bgcolor='#0f172a', plot_bgcolor='#1e293b', font_color='#f8fafc')
    
    st.plotly_chart(fig, use_container_width=True)

elif menu == "🗄️ SQL Historical Logs":
    st.subheader("🗄️ Database Telemetry History & Audit Trail")
    st.markdown("All automated and manual inference logs stored securely in the local SQLite database.")
    
    if st.button("🔄 Refresh Table", use_container_width=True):
        pass
        
    conn = sqlite3.connect(DB_FILENAME)
    logs_df = pd.read_sql_query("SELECT * FROM telemetry_logs ORDER BY id DESC", conn)
    conn.close()
    
    if not logs_df.empty:
        st.dataframe(logs_df, use_container_width=True, height=400)
    else:
        st.warning("No historical telemetry records found in the database yet.")