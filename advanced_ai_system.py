import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import pandas as pd
import os
import joblib
import sqlite3
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from datetime import datetime

# Matplotlib for embedded charts
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

MODEL_FILENAME = "ai_model.joblib"
DB_FILENAME = "enterprise_maintenance.db"

class AdvancedPredictiveSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("Enterprise AI & Predictive Maintenance Suite v6.0 (SQL Integrated)")
        self.root.geometry("1000x750")
        self.root.configure(bg="#0f172a")

        # Initialize SQLite Database
        self.init_database()

        # Style Configuration
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure("TNotebook", background="#0f172a", borderwidth=0)
        self.style.configure("TNotebook.Tab", font=("Segoe UI", 10, "bold"), padding=[12, 8])

        # Header Frame
        header_frame = tk.Frame(root, bg="#1e293b", height=70)
        header_frame.pack(fill=tk.X, padx=10, pady=10)
        header_frame.pack_propagate(False)

        tk.Label(header_frame, text=" ENTERPRISE AI & SQL ANALYTICS ", font=("Segoe UI", 14, "bold"), fg="#38bdf8", bg="#1e293b").pack(side=tk.LEFT, padx=15)
        self.model_status_lbl = tk.Label(header_frame, text="Model Status: Initializing...", font=("Segoe UI", 9, "bold"), fg="#fbbf24", bg="#1e293b")
        self.model_status_lbl.pack(side=tk.RIGHT, padx=15)

        # Notebook / Tabs
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.tab_predict = tk.Frame(self.notebook, bg="#f8fafc")
        self.tab_metrics = tk.Frame(self.notebook, bg="#f8fafc")
        self.tab_charts = tk.Frame(self.notebook, bg="#f8fafc")
        self.tab_logs = tk.Frame(self.notebook, bg="#f8fafc")

        self.notebook.add(self.tab_predict, text="⚡ Live AI Prediction")
        self.notebook.add(self.tab_metrics, text="📊 Performance Metrics")
        self.notebook.add(self.tab_charts, text="📈 Live Analytics Charts")
        self.notebook.add(self.tab_logs, text="💻 System Logs & SQL")

        # Build Tabs Content
        self.build_prediction_tab()
        self.build_metrics_tab()
        self.build_charts_tab()
        self.build_logs_tab()

        # Initialize, Load or Train Model
        self.load_or_train_model()

    def init_database(self):
        try:
            self.db_conn = sqlite3.connect(DB_FILENAME)
            self.db_cursor = self.db_conn.cursor()
            self.db_cursor.execute("""
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
            self.db_conn.commit()
        except Exception as e:
            print(f"Database Initialization Error: {str(e)}")

    def load_or_train_model(self):
        try:
            np.random.seed(42)
            n_samples = 1000
            self.temperatures = np.random.uniform(50.0, 110.0, n_samples)
            self.vibrations = np.random.uniform(0.5, 8.0, n_samples)
            self.pressures = np.random.uniform(20.0, 80.0, n_samples)
            
            failure_condition = (self.temperatures > 85.0) & (self.vibrations > 4.0) | (self.pressures > 65.0)
            failures = np.where(failure_condition, 1, 0)
            noise = np.random.binomial(1, 0.05, n_samples)
            self.failures = np.clip(failures + noise, 0, 1)

            df = pd.DataFrame({
                'Temperature': self.temperatures,
                'Vibration': self.vibrations,
                'Pressure': self.pressures,
                'Failure': self.failures
            })

            X = df[['Temperature', 'Vibration', 'Pressure']]
            y = df['Failure']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            if os.path.exists(MODEL_FILENAME):
                self.log_message(f"Loading pre-trained model from disk ({MODEL_FILENAME})...")
                self.ml_model = joblib.load(MODEL_FILENAME)
                self.log_message("Model successfully loaded from file!")
            else:
                self.log_message("Training Gradient Boosting Classifier...")
                self.ml_model = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, random_state=42)
                self.ml_model.fit(X_train, y_train)
                joblib.dump(self.ml_model, MODEL_FILENAME)
                self.log_message(f"Model trained and saved successfully to {MODEL_FILENAME}!")

            y_pred = self.ml_model.predict(X_test)
            self.accuracy = accuracy_score(y_test, y_pred) * 100
            self.report = classification_report(y_test, y_pred)

            self.model_status_lbl.config(text=f"Model Status: Active | DB: Connected", fg="#4ade80")
            self.log_message("SQLite database connected (`enterprise_maintenance.db`) successfully.")
            
            # Update Metrics Tab
            self.metrics_text.config(state=tk.NORMAL)
            self.metrics_text.delete("1.0", tk.END)
            self.metrics_text.insert(tk.END, f"=== MACHINE LEARNING & SQL EVALUATION ===\n\n")
            self.metrics_text.insert(tk.END, f"Algorithm: Gradient Boosting Classifier\n")
            self.metrics_text.insert(tk.END, f"Database Backend: SQLite3 ({DB_FILENAME})\n")
            self.metrics_text.insert(tk.END, f"Model Accuracy: {self.accuracy:.2f}%\n\n")
            self.metrics_text.insert(tk.END, "Classification Report:\n")
            self.metrics_text.insert(tk.END, self.report)
            self.metrics_text.config(state=tk.DISABLED)

            self.plot_analytics_chart()

        except Exception as e:
            self.log_message(f"[CRITICAL ERROR] {str(e)}")

    def build_prediction_tab(self):
        frame = tk.LabelFrame(self.tab_predict, text=" Real-time Sensor Telemetry Input ", font=("Segoe UI", 11, "bold"), bg="#f8fafc", fg="#334155", padx=20, pady=20)
        frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        tk.Label(frame, text="Core Temperature (°C):", font=("Segoe UI", 10, "bold"), bg="#f8fafc", fg="#475569").grid(row=0, column=0, sticky="w", pady=12)
        self.temp_entry = tk.Entry(frame, font=("Segoe UI", 11), width=25)
        self.temp_entry.grid(row=0, column=1, padx=15, pady=12)
        self.temp_entry.insert(0, "75.5")

        tk.Label(frame, text="Vibration Amplitude:", font=("Segoe UI", 10, "bold"), bg="#f8fafc", fg="#475569").grid(row=1, column=0, sticky="w", pady=12)
        self.vib_entry = tk.Entry(frame, font=("Segoe UI", 11), width=25)
        self.vib_entry.grid(row=1, column=1, padx=15, pady=12)
        self.vib_entry.insert(0, "2.1")

        tk.Label(frame, text="System Pressure (PSI):", font=("Segoe UI", 10, "bold"), bg="#f8fafc", fg="#475569").grid(row=2, column=0, sticky="w", pady=12)
        self.press_entry = tk.Entry(frame, font=("Segoe UI", 11), width=25)
        self.press_entry.grid(row=2, column=1, padx=15, pady=12)
        self.press_entry.insert(0, "45.0")

        btn_predict = tk.Button(frame, text="Execute AI Inference & Log to SQL", font=("Segoe UI", 11, "bold"), bg="#0284c7", fg="white", padx=20, pady=8, command=self.run_inference)
        btn_predict.grid(row=3, column=0, columnspan=2, pady=25)

        self.result_lbl = tk.Label(frame, text="System Status: Ready for Live Telemetry Analysis", font=("Segoe UI", 11, "bold"), fg="#0f172a", bg="#f8fafc")
        self.result_lbl.grid(row=4, column=0, columnspan=2, pady=10)

    def build_metrics_tab(self):
        self.metrics_text = tk.Text(self.tab_metrics, font=("Consolas", 10), bg="#0f172a", fg="#38bdf8", padx=15, pady=15)
        scrollbar = ttk.Scrollbar(self.tab_metrics, orient="vertical", command=self.metrics_text.yview)
        self.metrics_text.configure(yscrollcommand=scrollbar.set)
        
        self.metrics_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=10)
        self.metrics_text.config(state=tk.DISABLED)

    def build_charts_tab(self):
        self.chart_frame = tk.Frame(self.tab_charts, bg="#f8fafc")
        self.chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def plot_analytics_chart(self):
        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        
        ax.scatter(self.temperatures[:300], self.vibrations[:300], c=self.failures[:300], cmap='coolwarm', alpha=0.7, edgecolors='k')
        ax.set_title("Sensor Telemetry Distribution (Temp vs Vibration)", fontsize=10, fontweight='bold', color='#1e293b')
        ax.set_xlabel("Temperature (°C)", fontsize=9)
        ax.set_ylabel("Vibration Amplitude", fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.5)

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def build_logs_tab(self):
        top_frame = tk.Frame(self.tab_logs, bg="#f8fafc")
        top_frame.pack(fill=tk.X, padx=10, pady=5)

        btn_export = tk.Button(top_frame, text="📁 Export System Logs", font=("Segoe UI", 10, "bold"), bg="#059669", fg="white", padx=10, pady=6, command=self.export_logs)
        btn_export.pack(side=tk.LEFT, padx=5, pady=5)

        btn_db_history = tk.Button(top_frame, text="🗄️ Load SQL History", font=("Segoe UI", 10, "bold"), bg="#4f46e5", fg="white", padx=10, pady=6, command=self.load_db_history)
        btn_db_history.pack(side=tk.LEFT, padx=5, pady=5)

        self.log_text = tk.Text(self.tab_logs, font=("Consolas", 10), bg="#0f172a", fg="#4ade80", padx=15, pady=15)
        scrollbar = ttk.Scrollbar(self.tab_logs, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)

        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)

    def log_message(self, message):
        timestamp = datetime.now().strftime("[%H:%M:%S]")
        log_entry = f"{timestamp} {message}\n"
        try:
            self.log_text.insert(tk.END, log_entry)
            self.log_text.see(tk.END)
        except:
            pass

    def export_logs(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")], title="Save System Logs")
        if file_path:
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(self.log_text.get("1.0", tk.END))
                messagebox.showinfo("Success", f"Logs successfully exported to:\n{file_path}")
                self.log_message(f"System logs exported successfully to {file_path}")
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to save file: {str(e)}")

    def load_db_history(self):
        try:
            self.db_cursor.execute("SELECT * FROM telemetry_logs ORDER BY id DESC LIMIT 50")
            rows = self.db_cursor.fetchall()
            self.log_message("=== RECENT SQL DATABASE HISTORY (Last Records) ===")
            for row in rows:
                self.log_message(f"[SQL ID: {row[0]}] Time: {row[1]} | T:{row[2]} V:{row[3]} P:{row[4]} | Status: {row[5]} ({row[6]:.1f}%)")
            messagebox.showinfo("SQL Database", f"Successfully loaded {len(rows)} records from SQLite database into logs.")
        except Exception as e:
            messagebox.showerror("DB Error", f"Failed to query database: {str(e)}")

    def run_inference(self):
        try:
            temp = float(self.temp_entry.get())
            vib = float(self.vib_entry.get())
            press = float(self.press_entry.get())

            input_features = pd.DataFrame([[temp, vib, press]], columns=['Temperature', 'Vibration', 'Pressure'])
            prediction = self.ml_model.predict(input_features)
            probability = self.ml_model.predict_proba(input_features)[0]

            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            status_str = "CRITICAL" if prediction[0] == 1 else "NORMAL"
            conf_pct = float(probability[1] * 100 if prediction[0] == 1 else probability[0] * 100)

            # Insert into SQLite Database
            self.db_cursor.execute(
                "INSERT INTO telemetry_logs (timestamp, temperature, vibration, pressure, prediction_status, confidence) VALUES (?, ?, ?, ?, ?, ?)",
                (current_time, temp, vib, press, status_str, conf_pct)
            )
            self.db_conn.commit()

            if prediction[0] == 1:
                self.result_lbl.config(text=f"⚠️ CRITICAL ALERT: Failure Risk Detected! ({conf_pct:.1f}% Probability)", fg="#dc2626")
                self.log_message(f"SQL Logged [WARN]: T={temp}, V={vib}, P={press} -> Risk ({conf_pct:.1f}%)")
                messagebox.showwarning("AI Predictive Alert", f"High risk of hardware failure detected!\nConfidence: {conf_pct:.1f}%\nSaved to SQL Database.")
            else:
                self.result_lbl.config(text=f"✅ NORMAL: System Operating Safely ({conf_pct:.1f}% Confidence)", fg="#16a34a")
                self.log_message(f"SQL Logged [SAFE]: T={temp}, V={vib}, P={press} -> Normal ({conf_pct:.1f}%)")
                messagebox.showinfo("AI Status", f"System is operating normally.\nConfidence: {conf_pct:.1f}%\nSaved to SQL Database.")

        except ValueError:
            messagebox.showerror("Input Error", "Please ensure all sensor input fields contain valid numerical values!")
            self.log_message("[ERROR] Invalid user input during inference.")

if __name__ == "__main__":
    root = tk.Tk()
    app = AdvancedPredictiveSystem(root)
    root.mainloop()