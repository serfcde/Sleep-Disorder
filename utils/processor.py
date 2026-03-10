import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

class SleepAnalytics:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)

    def preprocess(self):
        """Advanced Feature Engineering & Signal Cleaning"""
        # 1. Decomposition of Blood Pressure (Vascular Load)
        if 'Blood Pressure' in self.df.columns:
            self.df[['Systolic', 'Diastolic']] = self.df['Blood Pressure'].str.split('/', expand=True).astype(int)
            self.df['Pulse_Pressure'] = self.df['Systolic'] - self.df['Diastolic']

        # 2. Cardiac Stress Index (CSI) - Ratio of Heart Rate to Sleep Duration
        self.df['Cardiac_Stress_Index'] = self.df['Heart Rate'] / self.df['Sleep Duration']

        # 3. Sleep Debt Calculation (Baseline 7.5 hours)
        self.df['Sleep_Debt'] = self.df['Sleep Duration'].apply(lambda x: max(0, 7.5 - x))

        # 4. Encoding BMI using Ordinal Mapping
        bmi_map = {'Normal': 0, 'Normal Weight': 0, 'Overweight': 1, 'Obese': 2}
        self.df['BMI_Score'] = self.df['BMI Category'].map(bmi_map)

        return self.df

    def detect_early_warnings(self):
        """Unsupervised Anomaly Detection for Deterioration Forecasting"""
        scaler = RobustScaler()
        features = ['Sleep Duration', 'Stress Level', 'Heart Rate', 'Sleep_Debt', 'Cardiac_Stress_Index']
        scaled_data = scaler.fit_transform(self.df[features])

        # Isolation Forest — kept for statistical outlier detection
        model = IsolationForest(n_estimators=100, contamination=0.12, random_state=42)
        self.df['Anomaly_Signal'] = model.fit_predict(scaled_data)

        # Separate readable column — used in dashboard table
        self.df['Statistical_Outlier'] = self.df['Anomaly_Signal'].map({-1: "⚠️ Outlier", 1: "Normal"})

        # --- FIXED RISK SCORE ---
        # HR only contributes if above healthy baseline of 65 bpm
        hr_excess    = (self.df['Heart Rate'] - 65).clip(lower=0)
        hr_score     = (hr_excess / 25) * 20        # max 20 pts at HR=90+

        stress_score = (self.df['Stress Level'] / 10) * 40   # max 40 pts
        sleep_score  = (self.df['Sleep_Debt'] / 3.5) * 40    # max 40 pts

        self.df['Risk_Score'] = (stress_score + sleep_score + hr_score).clip(0, 100).round(1)

        # --- CLEAN THRESHOLD-ONLY EARLY WARNING ---
        # Score >= 65 → CRITICAL (no anomaly gate, fully explainable)
        # Score >= 45 → MONITORING
        # Score <  45 → STABLE
        self.df['Early_Warning'] = np.where(
            self.df['Risk_Score'] >= 65,
            "🚨 CRITICAL DETERIORATION",
            np.where(self.df['Risk_Score'] >= 45, "⚠️ MONITORING", "✅ STABLE")
        )

        return self.df