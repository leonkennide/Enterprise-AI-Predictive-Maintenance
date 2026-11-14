import numpy as np
from sklearn.ensemble import RandomForestClassifier
import tkinter as tk
from tkinter import messagebox

# 1. محاكاة تدريب نموذج تعلم الآلة (Machine Learning Model Training)
# نفترض أن لدينا بيانات تدريب سابقة لـ 5 آلات تشمل: [درجة الحرارة، مستوى الاهتزاز، الضغط]
# والنتيجة: 0 تعني (الآلة سليمة)، 1 تعني (معرضة للخطر/ستتعطل)
X_train = np.array([
    [65.0, 1.2, 30.0],  # سليم
    [85.5, 4.5, 55.2],  # خطر
    [70.1, 1.8, 32.1],  # سليم
    [95.2, 5.8, 60.0],  # خطر
    [62.0, 1.0, 29.5],  # سليم
    [88.0, 4.9, 58.1]   # خطر
])
y_train = np.array([0, 1, 0, 1, 0, 1])

# تدريب نموذج الغابات العشوائية (Random Forest Classifier)
ml_model = RandomForestClassifier(n_estimators=100, random_state=42)
ml_model.fit(X_train, y_train)


# 2. بناء واجهة مستخدم بسيطة واحترافية باستخدام tkinter
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

        # مدخل درجة الحرارة
        tk.Label(form_frame, text="Temperature (°C):", fg="white", bg="#1e293b", font=("Arial", 10)).grid(row=0, column=0, sticky="w", pady=8)
        self.temp_entry = tk.Entry(form_frame, font=("Arial", 11), width=15)
        self.temp_entry.grid(row=0, column=1, padx=10, pady=8)

        # مدخل مستوى الاهتزاز
        tk.Label(form_frame, text="Vibration Level:", fg="white", bg="#1e293b", font=("Arial", 10)).grid(row=1, column=0, sticky="w", pady=8)
        self.vib_entry = tk.Entry(form_frame, font=("Arial", 11), width=15)
        self.vib_entry.grid(row=1, column=1, padx=10, pady=8)

        # مدخل الضغط
        tk.Label(form_frame, text="Pressure (PSI):", fg="white", bg="#1e293b", font=("Arial", 10)).grid(row=2, column=0, sticky="w", pady=8)
        self.press_entry = tk.Entry(form_frame, font=("Arial", 11), width=15)
        self.press_entry.grid(row=2, column=1, padx=10, pady=8)

        # زر التنبؤ بالذكاء الاصطناعي
        predict_btn = tk.Button(root, text="Run AI Prediction", bg="#059669", fg="white", font=("Arial", 11, "bold"), padx=15, pady=8, command=self.predict_failure)
        predict_btn.pack(pady=15)

    def predict_failure(self):
        try:
            # قراءة القيم المدخلة من المستخدم
            temp = float(self.temp_entry.get())
            vib = float(self.vib_entry.get())
            press = float(self.press_entry.get())

            # إدخال البيانات للنموذج للتنبؤ
            input_data = np.array([[temp, vib, press]])
            prediction = ml_model.predict(input_data)

            # عرض النتيجة بناءً على توقع الذكاء الاصطناعي
            if prediction[0] == 1:
                messagebox.showwarning("AI Alert", "⚠️ تحذير: النموذج يتوقع احتمال تعطل الآلة قريباً! يجب إجراء صيانة.")
            else:
                messagebox.showinfo("AI Status", "✅ الآلة تعمل بشكل طبيعي ولا توجد مخاطر.")

        except ValueError:
            messagebox.showerror("Error", "الرجاء إدخال أرقام صحيحة في جميع الحقول!")

if __name__ == "__main__":
    root = tk.Tk()
    app = PredictorApp(root)
    root.mainloop()