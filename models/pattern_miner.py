from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

class BehavioralMiner:

    def __init__(self, dataframe):

        self.df = dataframe

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

    def find_optimal_clusters(self):

        X = self.scaler.fit_transform(self.df[self.features])

        scores = {}
        for k in range(2,7):

            model = KMeans(n_clusters=k, n_init=20, random_state=42)
            labels = model.fit_predict(X)

            score = silhouette_score(X, labels)
            scores[k] = score

        best_k = max(scores, key=scores.get)

        return best_k, scores

    def run_mining(self, n_clusters=None):

        X = self.scaler.fit_transform(self.df[self.features])

        if n_clusters is None:
            n_clusters,_ = self.find_optimal_clusters()

        self.model = KMeans(n_clusters=n_clusters,n_init=20,max_iter=500,random_state=42)

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
            (100 - self.df['Sleep Duration']*10)/10
        )

        self.df['Risk Score'] = risk_score

        self.df['Risk Level'] = pd.cut(
            risk_score,
            bins=[0,5,10,20],
            labels=['Low Risk','Moderate Risk','High Risk']
        )

        return self.df