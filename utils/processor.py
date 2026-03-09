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
        # We use RobustScaler to handle outliers in health data
        scaler = RobustScaler()
        features = ['Sleep Duration', 'Stress Level', 'Heart Rate', 'Sleep_Debt', 'Cardiac_Stress_Index']
        scaled_data = scaler.fit_transform(self.df[features])
        
        # Isolation Forest: High sensitivity to detect subtle health shifts
        model = IsolationForest(n_estimators=100, contamination=0.12, random_state=42)
        self.df['Anomaly_Signal'] = model.fit_predict(scaled_data)
        
        # Risk Scoring Logic: 0 (Safe) to 100 (Critical)
        # Based on Sleep Duration degradation and Stress elevation
        self.df['Risk_Score'] = (
            (self.df['Stress Level'] * 10) + 
            (self.df['Sleep_Debt'] * 15) + 
            (self.df['Heart Rate'] * 0.5)
        ).clip(0, 100)
        
        # Deterioration Trigger
        self.df['Early_Warning'] = np.where(
            (self.df['Anomaly_Signal'] == -1) & (self.df['Risk_Score'] > 60),
            "CRITICAL: Deterioration Detected",
            np.where(self.df['Risk_Score'] > 40, "WARNING: Monitor Habits", "Stable")
        )
        return self.df