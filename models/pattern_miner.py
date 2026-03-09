from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import pandas as pd

class BehavioralMiner:
    def __init__(self, dataframe):
        self.df = dataframe
        self.features = ['Sleep Duration', 'Quality of Sleep', 'Stress Level', 'Heart Rate', 'Physical Activity Level']
        self.scaler = StandardScaler()
        self.model = None

    def run_mining(self, n_clusters=3):
        """Executes K-Means and returns metrics for academic validation"""
        X = self.scaler.fit_transform(self.df[self.features])
        
        self.model = KMeans(n_clusters=n_clusters, n_init=20, max_iter=500, random_state=42)
        self.df['Behavioral_Cluster'] = self.model.fit_predict(X)
        
        # Validation Metric (Critical for 30-mark projects)
        score = silhouette_score(X, self.df['Behavioral_Cluster'])
        return self.df, score

    def get_cluster_profiles(self):
        """Interprets the 'Meaning' of each cluster"""
        # Calculate means and rename for clarity
        profile = self.df.groupby('Behavioral_Cluster')[self.features].mean()
        profile.index = [f"Pattern {i}: " + ("High Risk" if i==2 else "Moderate" if i==1 else "Optimal") for i in profile.index]
        return profile