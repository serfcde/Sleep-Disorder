from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

class BehavioralMiner:
    def __init__(self, dataframe):
        self.df = dataframe.copy()
        self.features = [
            'Sleep Duration',
            'Quality of Sleep',
            'Stress Level',
            'Heart Rate',
            'Physical Activity Level',
            'Daily Steps',
            'Age'
        ]
        self.scaler = StandardScaler()
        self.model = None
        self._clean_features()

    def _clean_features(self):
        """Keep clustering stable when the source data has blanks or typed strings."""
        self.df.columns = self.df.columns.str.strip()
        missing_features = [feature for feature in self.features if feature not in self.df.columns]
        if missing_features:
            raise ValueError(f"Missing required clustering features: {missing_features}")

        for feature in self.features:
            self.df[feature] = pd.to_numeric(self.df[feature], errors='coerce')
            self.df[feature] = self.df[feature].fillna(self.df[feature].median())

    def find_optimal_clusters(self):
        X = self.scaler.fit_transform(self.df[self.features])
        scores = {}
        max_k = min(6, len(self.df) - 1)
        for k in range(2, max_k + 1):
            model = KMeans(n_clusters=k, n_init=30, max_iter=500, random_state=42)
            labels = model.fit_predict(X)
            score = silhouette_score(X, labels)
            scores[k] = score
        best_k = max(scores, key=scores.get) if scores else 2
        return best_k, scores

    def run_mining(self, n_clusters=None):
        X = self.scaler.fit_transform(self.df[self.features])
        if n_clusters is None:
            n_clusters,_ = self.find_optimal_clusters()
        self.model = KMeans(n_clusters=n_clusters,n_init=30,max_iter=500,random_state=42)
        self.df['Behavioral_Cluster'] = self.model.fit_predict(X)
        score = silhouette_score(X,self.df['Behavioral_Cluster'])
        return self.df,score

    def get_cluster_profiles(self):
        profile = self.df.groupby('Behavioral_Cluster')[self.features].mean()
        return profile

    def risk_scoring(self):
        risk_score = (
            (10 - self.df['Quality of Sleep']) +
            self.df['Stress Level'] +
            (7.5 - self.df['Sleep Duration']).clip(lower=0) * 2
        )
        self.df['Risk Score'] = risk_score
        self.df['Risk Level'] = pd.cut(
            risk_score,
            bins=[-np.inf,5,10,np.inf],
            labels=['Low Risk','Moderate Risk','High Risk']
        )
        return self.df
