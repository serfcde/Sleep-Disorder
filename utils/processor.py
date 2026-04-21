import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

class SleepAnalytics:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)

    def preprocess(self):
        """Advanced Feature Engineering & Signal Cleaning"""
        self.df.columns = self.df.columns.str.strip()
        self.df = self.df.dropna(how='all')

        text_cols = self.df.select_dtypes(include='object').columns
        for col in text_cols:
            self.df[col] = self.df[col].astype(str).str.strip()
            self.df[col] = self.df[col].replace({'': np.nan, 'nan': np.nan})

        if 'Sleep Disorder' in self.df.columns:
            self.df['Sleep Disorder'] = self.df['Sleep Disorder'].fillna('None')

        # 1. Decomposition of Blood Pressure (Vascular Load)
        if 'Blood Pressure' in self.df.columns:
            bp = self.df['Blood Pressure'].astype(str).str.extract(r'(?P<Systolic>\d+)\s*/\s*(?P<Diastolic>\d+)')
            self.df['Systolic'] = pd.to_numeric(bp['Systolic'], errors='coerce')
            self.df['Diastolic'] = pd.to_numeric(bp['Diastolic'], errors='coerce')
            self.df['Pulse_Pressure'] = self.df['Systolic'] - self.df['Diastolic']

        numerical_cols = ['Sleep Duration', 'Stress Level', 'Heart Rate', 'Systolic', 'Diastolic',
                          'Pulse_Pressure', 'Quality of Sleep', 'Physical Activity Level', 'Daily Steps', 'Age']
        for col in [c for c in numerical_cols if c in self.df.columns]:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            self.df[col] = self.df[col].fillna(self.df[col].median())

        # 2. Cardiac Stress Index (CSI) - Ratio of Heart Rate to Sleep Duration
        self.df['Cardiac_Stress_Index'] = self.df['Heart Rate'] / self.df['Sleep Duration'].replace(0, np.nan)
        self.df['Cardiac_Stress_Index'] = self.df['Cardiac_Stress_Index'].replace([np.inf, -np.inf], np.nan)
        self.df['Cardiac_Stress_Index'] = self.df['Cardiac_Stress_Index'].fillna(self.df['Cardiac_Stress_Index'].median())

        # 3. Sleep Debt Calculation (Baseline 7.5 hours)
        self.df['Sleep_Debt'] = (7.5 - self.df['Sleep Duration']).clip(lower=0)

        # 4. Encoding BMI using Ordinal Mapping
        bmi_map = {'Normal': 0, 'Normal Weight': 0, 'Overweight': 1, 'Obese': 2}
        self.df['BMI_Score'] = self.df['BMI Category'].map(bmi_map).fillna(0)

        # 5. Robust Normalization to Handle Outliers without overwriting readable values
        scaler = RobustScaler()
        scaled_cols = ['Sleep Duration', 'Stress Level', 'Heart Rate', 'Systolic', 'Diastolic', 'Pulse_Pressure',
                       'Cardiac_Stress_Index', 'Sleep_Debt', 'BMI_Score']
        scaled_cols = [col for col in scaled_cols if col in self.df.columns]
        scaled_values = scaler.fit_transform(self.df[scaled_cols])
        for idx, col in enumerate(scaled_cols):
            self.df[f'{col}_Scaled'] = scaled_values[:, idx]

        return self.df

    def detect_early_warnings(self):
        """Unsupervised Anomaly Detection for Deterioration Forecasting"""
        scaler = RobustScaler()
        if 'Sleep_Debt' not in self.df.columns or 'Cardiac_Stress_Index' not in self.df.columns:
            self.preprocess()

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
        self.df['Early_Warning'] = np.where(
            self.df['Risk_Score'] >= 65,
            "🚨 CRITICAL DETERIORATION",
            np.where(self.df['Risk_Score'] >= 45, "⚠️ MONITORING", "✅ STABLE")
        )

        return self.df
