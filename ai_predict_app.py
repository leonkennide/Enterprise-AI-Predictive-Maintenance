import numpy as np
from sklearn.ensemble import RandomForestClassifier
import tkinter as tk
from tkinter import messagebox

# 1. Simulate Machine Learning Model Training
# Training dataset for 6 machines: [Temperature, Vibration Level, Pressure]
# Result: 0 means (Normal/Healthy), 1 means (High Risk / Failure Predicted)
X_train = np.array([
    [65.0, 1.2, 30.0],  # Healthy
    [85.5, 4.5, 55.2],  # Risk
    [70.1, 1.8, 32.1],  # Healthy
    [95.2, 5.8, 60.0],  # Risk
    [62.0, 1.0, 29.5],  # Healthy
    [88.0, 4.9, 58.1]   # Risk
])
y_train = np.array([0, 1, 0, 1, 0, 1])

# Train Random Forest Classifier
ml_model = RandomForestClassifier(n_estimators=100, random_state=42)
ml_model.fit(X_train, y_train)


# 2. Build Professional GUI using tkinter
class PredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Predictive Maintenance System")
        self.root.geometry("450x400")
        self.root.configure(bg="#0f172a")

        title_lbl = tk.Label(root, text="AI Machine Failure Predictor", font=("Arial", 14, "bold"), fg="#38bdf8", bg="#0f172a")
        title_lbl.pack(pady=15)

        form_frame = tk.Frame(root, bg="#1e293b", padx=15, pady=15)
        form_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Temperature Input
        tk.Label(form_frame, text="Temperature (°C):", fg="white", bg="#1e293b", font=("Arial", 10)).grid(row=0, column=0, sticky="w", pady=8)
        self.temp_entry = tk.Entry(form_frame, font=("Arial", 11), width=15)
        self.temp_entry.grid(row=0, column=1, padx=10, pady=8)

        # Vibration Level Input
        tk.Label(form_frame, text="Vibration Level:", fg="white", bg="#1e293b", font=("Arial", 10)).grid(row=1, column=0, sticky="w", pady=8)
        self.vib_entry = tk.Entry(form_frame, font=("Arial", 11), width=15)
        self.vib_entry.grid(row=1, column=1, padx=10, pady=8)

        # Pressure Input
        tk.Label(form_frame, text="Pressure (PSI):", fg="white", bg="#1e293b", font=("Arial", 10)).grid(row=2, column=0, sticky="w", pady=8)
        self.press_entry = tk.Entry(form_frame, font=("Arial", 11), width=15)
        self.press_entry.grid(row=2, column=1, padx=10, pady=8)

        # Run Prediction Button
        predict_btn = tk.Button(root, text="Run AI Prediction", bg="#059669", fg="white", font=("Arial", 11, "bold"), padx=15, pady=8, command=self.predict_failure)
        predict_btn.pack(pady=15)

    def predict_failure(self):
        try:
            # Read input values from user
            temp = float(self.temp_entry.get())
            vib = float(self.vib_entry.get())
            press = float(self.press_entry.get())

            # Pass data to the ML model for prediction
            input_data = np.array([[temp, vib, press]])
            prediction = ml_model.predict(input_data)

            # Display result based on AI prediction
            if prediction[0] == 1:
                messagebox.showwarning("AI Alert", "⚠️ WARNING: The model predicts potential machine failure soon! Maintenance required.")
            else:
                messagebox.showinfo("AI Status", "✅ Status Normal: The machine is operating safely with no detected risks.")

        except ValueError:
            messagebox.showerror("Error", "Please enter valid numeric values in all fields!")

if __name__ == "__main__":
    root = tk.Tk()
    app = PredictorApp(root)
    root.mainloop()