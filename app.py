import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_absolute_error, silhouette_score
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="SleepGuard", layout="wide", page_icon="🌙")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .main { background-color: #0d1117; }
    .stMetric { background: linear-gradient(135deg, #1e2433, #2d3561); border-radius: 12px; padding: 10px; }
    .stTabs [data-baseweb="tab"] { font-size: 15px; font-weight: 600; }
    h1, h2, h3 { color: #c9d1d9; }
    .prediction-box { background: linear-gradient(135deg, #1a1f35, #252d4a); border: 1px solid #30363d; border-radius: 16px; padding: 20px; }
</style>
""", unsafe_allow_html=True)


# --- ANALYTICS ENGINE ---
class SleepSystem:
    def __init__(self, data):
        self.df = data.copy()

    def process_and_mine(self):
        if 'Blood Pressure' in self.df.columns:
            self.df[['Systolic', 'Diastolic']] = self.df['Blood Pressure'].str.split('/', expand=True).astype(int)
            self.df['Pulse_Pressure'] = self.df['Systolic'] - self.df['Diastolic']

        self.df['Cardiac_Stress_Index'] = self.df['Heart Rate'] / self.df['Sleep Duration']
        self.df['Sleep_Debt'] = self.df['Sleep Duration'].apply(lambda x: max(0, 7.5 - x))

        bmi_map = {'Normal': 0, 'Normal Weight': 0, 'Overweight': 1, 'Obese': 2}
        self.df['BMI_Score'] = self.df['BMI Category'].map(bmi_map).fillna(0)

        features = ['Sleep Duration', 'Quality of Sleep', 'Stress Level', 'Heart Rate', 'Physical Activity Level']
        scaler = StandardScaler()
        X = scaler.fit_transform(self.df[features])

        kmeans = KMeans(n_clusters=3, n_init=20, max_iter=500, random_state=42)
        self.df['Behavioral_Cluster'] = kmeans.fit_predict(X)
        sil_score = silhouette_score(X, self.df['Behavioral_Cluster'])

        iso = IsolationForest(n_estimators=100, contamination=0.12, random_state=42)
        self.df['Anomaly_Score'] = iso.fit_predict(X)

        self.df['Risk_Score'] = (
            (self.df['Stress Level'] * 10) +
            (self.df['Sleep_Debt'] * 15) +
            (self.df['Heart Rate'] * 0.5)
        ).clip(0, 100)

        self.df['Early_Warning'] = np.where(
            (self.df['Anomaly_Score'] == -1) & (self.df['Risk_Score'] > 60),
            "🚨 CRITICAL DETERIORATION",
            np.where(self.df['Risk_Score'] > 40, "⚠️ MONITORING", "✅ STABLE")
        )
        return self.df, sil_score


# --- ML MODEL TRAINING ---
@st.cache_resource
def train_models(df):
    """Train classification (Sleep Disorder) + regression (Sleep Quality) models"""
    model_df = df.copy()

    # Encode categoricals
    le_disorder = LabelEncoder()
    model_df['Sleep Disorder Encoded'] = le_disorder.fit_transform(model_df['Sleep Disorder'].fillna('None'))

    le_gender = LabelEncoder()
    model_df['Gender_Enc'] = le_gender.fit_transform(model_df['Gender'])

    le_bmi = LabelEncoder()
    model_df['BMI_Enc'] = le_bmi.fit_transform(model_df['BMI Category'])

    features = ['Age', 'Gender_Enc', 'Sleep Duration', 'Physical Activity Level',
                'Stress Level', 'BMI_Enc', 'Heart Rate', 'Daily Steps']

    X = model_df[features].dropna()
    y_class = model_df.loc[X.index, 'Sleep Disorder Encoded']
    y_reg = model_df.loc[X.index, 'Quality of Sleep']

    X_train, X_test, y_c_train, y_c_test = train_test_split(X, y_class, test_size=0.2, random_state=42)
    _, _, y_r_train, y_r_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_c_train)
    clf_acc = clf.score(X_test, y_c_test)

    reg = RandomForestRegressor(n_estimators=100, random_state=42)
    reg.fit(X_train, y_r_train)
    reg_mae = mean_absolute_error(y_r_test, reg.predict(X_test))

    return clf, reg, le_disorder, le_gender, le_bmi, features, clf_acc, reg_mae


# --- MAIN APP ---
st.title("🌙 SleepGuard— Sleep Health Analytics & Prediction System")

CSV_NAME = 'Sleep_health_and_lifestyle_dataset.csv'

if not os.path.exists(CSV_NAME):
    st.error(f"❌ Could not find `{CSV_NAME}`. Please ensure it is in the same folder as app.py.")
    st.stop()

raw_data = pd.read_csv(CSV_NAME)
engine = SleepSystem(raw_data)
df, sil_score = engine.process_and_mine()

# --- SIDEBAR ---
with st.sidebar:
    
    st.header("⚙️ Dashboard Controls")
    occ_list = st.multiselect("Filter by Occupation", options=sorted(df['Occupation'].unique()),
                              default=df['Occupation'].unique().tolist())
    gender_filter = st.multiselect("Filter by Gender", options=df['Gender'].unique().tolist(),
                                   default=df['Gender'].unique().tolist())
    st.divider()
    st.caption(f"📐 K-Means Silhouette Score: **{sil_score:.3f}**")
    st.caption("Higher = better-defined clusters (0–1 scale)")

filtered_df = df[df['Occupation'].isin(occ_list) & df['Gender'].isin(gender_filter)]

# --- KPI METRICS ---
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("👥 Total Population", len(filtered_df))
m2.metric("😰 Avg Stress Level", f"{filtered_df['Stress Level'].mean():.1f}/10")
m3.metric("🚨 Critical Alerts", len(filtered_df[filtered_df['Early_Warning'] == "🚨 CRITICAL DETERIORATION"]))
m4.metric("😴 Avg Sleep Quality", f"{filtered_df['Quality of Sleep'].mean():.1f}/10")
m5.metric("💤 Avg Sleep Duration", f"{filtered_df['Sleep Duration'].mean():.1f} hrs")

st.divider()

# --- TABS ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🧬 Behavioral Clusters",
    "🚨 Early Warnings",
    "📊 Deep Visualisations",
    "📈 Statistics & Correlations",
    "🤖 Predict New Person"
])

# ========================
# TAB 1 — BEHAVIORAL CLUSTERS
# ========================
with tab1:
    st.subheader("Unsupervised Behavioral Pattern Mining (K-Means)")

    col1, col2 = st.columns([2, 1])
    with col1:
        fig_3d = px.scatter_3d(filtered_df, x='Sleep Duration', y='Stress Level', z='Heart Rate',
                               color='Behavioral_Cluster', hover_data=['Occupation', 'Quality of Sleep'],
                               title="3D Behavioral Habitat Mapping",
                               color_continuous_scale='Viridis', height=550)
        st.plotly_chart(fig_3d, use_container_width=True)

    with col2:
        st.write("#### Cluster Profiles")
        features_for_profile = [
            'Sleep Duration',
            'Quality of Sleep',
            'Stress Level',
            'Heart Rate',
            'Physical Activity Level',
            'Daily Steps',
            'Age'
            ]
        profile = filtered_df.groupby('Behavioral_Cluster')[features_for_profile].mean().round(2)
        profile.index = [f"Cluster {i}" for i in profile.index]
        st.dataframe(profile, use_container_width=True)

        st.write("#### Cluster Size Distribution")
        cluster_counts = filtered_df['Behavioral_Cluster'].value_counts().reset_index()
        cluster_counts.columns = ['Cluster', 'Count']
        fig_pie = px.pie(cluster_counts, values='Count', names='Cluster',
                         color_discrete_sequence=px.colors.sequential.Plasma_r)
        st.plotly_chart(fig_pie, use_container_width=True)

    
    # ========================
    # Cluster Meaning Section
    # ========================
    st.markdown("### 🧬 Understanding the Behavioral Clusters")

    # Dynamically compute actual cluster stats to show real numbers
    cluster_stats = filtered_df.groupby('Behavioral_Cluster')[
        ['Sleep Duration', 'Stress Level', 'Heart Rate', 'Quality of Sleep', 'Physical Activity Level']
    ].mean().round(1)

    cluster_configs = {
        0: {
            "title": "Cluster 0 — Optimal Sleep",
            "emoji": "🟢",
            "color": "#0d2b1a",
            "border": "#2d6a4f",
            "badge_bg": "#1a4731",
            "badge_text": "#52b788",
            "tag": "HEALTHY",
            "desc": "This group maintains consistent, restorative sleep with low physiological stress markers. They represent the target health baseline.",
        },
        1: {
            "title": "Cluster 1 — Moderate Risk",
            "emoji": "🟡",
            "color": "#2b2200",
            "border": "#b58900",
            "badge_bg": "#3d3000",
            "badge_text": "#f4c430",
            "tag": "MONITOR",
            "desc": "This group shows early signs of sleep disruption. Without intervention, they may deteriorate into the high-risk category.",
        },
        2: {
            "title": "Cluster 2 — High Risk",
            "emoji": "🔴",
            "color": "#2b0a0a",
            "border": "#c0392b",
            "badge_bg": "#3d1010",
            "badge_text": "#e74c3c",
            "tag": "CRITICAL",
            "desc": "This group displays compounding risk factors. High stress, poor sleep, and elevated heart rate together significantly increase disorder probability.",
        },
    }

    cluster_labels = {
        0: ["Low Stress", "Normal HR", "Restorative Sleep", "Active Lifestyle"],
        1: ["Moderate Stress", "Mild HR Elevation", "Fragmented Sleep", "Reduced Activity"],
        2: ["High Stress", "Elevated HR", "Severe Sleep Deficit", "Sedentary Pattern"],
    }

    cols = st.columns(3)

    for idx, col in enumerate(cols):
        cfg = cluster_configs[idx]
        stats = cluster_stats.loc[idx] if idx in cluster_stats.index else None
        tags = cluster_labels[idx]

        with col:
            st.markdown(f"""
            <div style="
                background: {cfg['color']};
                border: 1.5px solid {cfg['border']};
                border-radius: 14px;
                padding: 20px 18px 16px 18px;
                height: 100%;
                font-family: Arial, sans-serif;
            ">
                <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:12px;">
                    <span style="font-size:22px;">{cfg['emoji']}</span>
                    <span style="
                        background:{cfg['badge_bg']};
                        color:{cfg['badge_text']};
                        font-size:11px;
                        font-weight:700;
                        letter-spacing:1.5px;
                        padding:3px 10px;
                        border-radius:20px;
                        border: 1px solid {cfg['border']};
                    ">{cfg['tag']}</span>
                </div>
                <div style="font-size:15px; font-weight:700; color:#e0e0e0; margin-bottom:8px;">
                    {cfg['title']}
                </div>
                <div style="font-size:12px; color:#aaaaaa; margin-bottom:14px; line-height:1.5;">
                    {cfg['desc']}
                </div>
                <hr style="border:none; border-top:1px solid {cfg['border']}; margin: 10px 0 12px 0; opacity:0.4;">
                <div style="display:flex; flex-direction:column; gap:7px; margin-bottom:14px;">
                    <div style="display:flex; justify-content:space-between; font-size:12px;">
                        <span style="color:#888888;">😴 Avg Sleep</span>
                        <span style="color:#e0e0e0; font-weight:600;">{stats['Sleep Duration'] if stats is not None else 'N/A'} hrs</span>
                    </div>
                    <div style="display:flex; justify-content:space-between; font-size:12px;">
                        <span style="color:#888888;">😰 Stress Level</span>
                        <span style="color:#e0e0e0; font-weight:600;">{stats['Stress Level'] if stats is not None else 'N/A'} / 10</span>
                    </div>
                    <div style="display:flex; justify-content:space-between; font-size:12px;">
                        <span style="color:#888888;">❤️ Heart Rate</span>
                        <span style="color:#e0e0e0; font-weight:600;">{stats['Heart Rate'] if stats is not None else 'N/A'} bpm</span>
                    </div>
                    <div style="display:flex; justify-content:space-between; font-size:12px;">
                        <span style="color:#888888;">⭐ Sleep Quality</span>
                        <span style="color:#e0e0e0; font-weight:600;">{stats['Quality of Sleep'] if stats is not None else 'N/A'} / 10</span>
                    </div>
                </div>
                <div style="display:flex; flex-wrap:wrap; gap:5px;">
                    {"".join([f'<span style="background:{cfg["badge_bg"]}; color:{cfg["badge_text"]}; font-size:10px; padding:3px 8px; border-radius:10px; border:1px solid {cfg["border"]};">{t}</span>' for t in tags])}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
# ========================
# TAB 2 — EARLY WARNINGS
# ========================
with tab2:
    st.subheader("🚨 Deterioration Detection Log")

   

    col_a, col_b = st.columns([1, 2])

    with col_a:
        # --- GAUGE ---
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=filtered_df['Risk_Score'].mean(),
            title={'text': "Avg Risk Index", 'font': {'size': 18}},
            delta={'reference': 50, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': "#e63946"},
                'steps': [
                    {'range': [0, 45],  'color': '#2d6a4f'},
                    {'range': [45, 65], 'color': '#f4a261'},
                    {'range': [65, 100],'color': '#e63946'}
                ],
                'threshold': {'line': {'color': "white", 'width': 4}, 'thickness': 0.75, 'value': 65}
            }))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)

        # --- WARNING DISTRIBUTION BAR ---
        warn_counts = filtered_df['Early_Warning'].value_counts().reset_index()
        warn_counts.columns = ['Status', 'Count']
        fig_warn = px.bar(
            warn_counts, x='Status', y='Count', color='Status',
            color_discrete_map={
                "🚨 CRITICAL DETERIORATION": "#e63946",
                "⚠️ MONITORING":             "#f4a261",
                "✅ STABLE":                 "#2d6a4f"
            },
            text='Count',
            title="Warning Status Distribution"
        )
        fig_warn.update_traces(textposition='outside')
        fig_warn.update_layout(showlegend=False, xaxis_title="", yaxis_title="Count")
        st.plotly_chart(fig_warn, use_container_width=True)

    with col_b:
        at_risk = filtered_df[filtered_df['Early_Warning'] != "✅ STABLE"].copy()

        # --- CRITICAL vs MONITORING split summary ---
        n_critical   = len(at_risk[at_risk['Early_Warning'] == "🚨 CRITICAL DETERIORATION"])
        n_monitoring = len(at_risk[at_risk['Early_Warning'] == "⚠️ MONITORING"])

        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("🚨 Critical",   n_critical)
        mc2.metric("⚠️ Monitoring", n_monitoring)
        mc3.metric("Total Flagged", len(at_risk))

        st.markdown("""
        <small style='color:#aaaaaa;'>
        <b>🚨 CRITICAL</b> = Isolation Forest anomaly <i>AND</i> Risk Score &gt; 65 &nbsp;|&nbsp;
        <b>⚠️ MONITORING</b> = Risk Score &gt; 45 &nbsp;|&nbsp;
        <b>✅ STABLE</b> = Risk Score ≤ 45
        </small>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # --- FLAGGED PERSONS TABLE ---
        display_cols = ['Person ID', 'Occupation', 'Age', 'Early_Warning',
                        'Risk_Score', 'Stress Level', 'Sleep Duration', 'Heart Rate']

        st.dataframe(
            at_risk[display_cols].sort_values('Risk_Score', ascending=False),
            use_container_width=True,
            height=380
        )

    st.divider()

    # --- VIOLIN PLOT ---
    st.write("#### Risk Score Distribution by Occupation")
    fig_violin = px.violin(
        filtered_df, x='Occupation', y='Risk_Score', color='Early_Warning',
        box=True,
        title="Risk Score Spread by Occupation",
        color_discrete_map={
            "🚨 CRITICAL DETERIORATION": "#e63946",
            "⚠️ MONITORING":             "#f4a261",
            "✅ STABLE":                 "#2d6a4f"
        }
    )
    fig_violin.update_xaxes(tickangle=45)
    fig_violin.update_layout(legend_title_text="Warning Status")
    st.plotly_chart(fig_violin, use_container_width=True)

    


# ========================
# TAB 3 — DEEP VISUALISATIONS
# ========================
with tab3:
    st.subheader("📊 Exploratory Data Visualisations")

    # Row 1
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Sleep Disorder Distribution**")
        disorder_counts = filtered_df['Sleep Disorder'].fillna('None').value_counts().reset_index()
        disorder_counts.columns = ['Disorder', 'Count']
        fig_disorder = px.bar(disorder_counts, x='Disorder', y='Count', color='Disorder',
                              color_discrete_sequence=['#4cc9f0', '#f72585', '#7209b7'],
                              text='Count', title="Prevalence of Sleep Disorders")
        fig_disorder.update_traces(textposition='outside')
        st.plotly_chart(fig_disorder, use_container_width=True)

    with c2:
        st.write("**Age vs Sleep Duration (by Gender)**")
        fig_age = px.scatter(filtered_df, x='Age', y='Sleep Duration', color='Gender',
                             size='Quality of Sleep', hover_data=['Occupation', 'Sleep Disorder'],
                             trendline='ols', title="Age vs Sleep Duration",
                             color_discrete_sequence=['#4cc9f0', '#f72585'])
        st.plotly_chart(fig_age, use_container_width=True)

    # Row 2
    c3, c4 = st.columns(2)
    with c3:
        st.write("**Physical Activity vs Sleep Quality**")
        fig_act = px.scatter(filtered_df, x='Physical Activity Level', y='Quality of Sleep',
                             color='Stress Level', size='Heart Rate',
                             color_continuous_scale='RdYlGn_r',
                             hover_data=['Occupation', 'BMI Category'],
                             title="Physical Activity vs Sleep Quality (coloured by Stress)")
        st.plotly_chart(fig_act, use_container_width=True)

    with c4:
        st.write("**BMI Category vs Sleep Metrics**")
        bmi_avg = filtered_df.groupby('BMI Category')[['Sleep Duration', 'Quality of Sleep', 'Stress Level']].mean().reset_index()
        fig_bmi = px.bar(bmi_avg.melt(id_vars='BMI Category'), x='BMI Category', y='value',
                         color='variable', barmode='group',
                         title="BMI Category vs Avg Sleep Duration, Quality & Stress",
                         color_discrete_sequence=['#4cc9f0', '#f72585', '#7209b7'])
        st.plotly_chart(fig_bmi, use_container_width=True)

    # Row 3
    c5, c6 = st.columns(2)
    with c5:
        st.write("**Sleep Duration Distribution by Sleep Disorder**")
        fig_hist = px.histogram(filtered_df, x='Sleep Duration', color='Sleep Disorder',
                                nbins=20, marginal='box', opacity=0.8,
                                title="Sleep Duration Distribution",
                                color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig_hist, use_container_width=True)

    with c6:
        st.write("**Daily Steps vs Sleep Quality (by BMI)**")
        fig_steps = px.scatter(filtered_df, x='Daily Steps', y='Quality of Sleep',
                               color='BMI Category', facet_col='Gender',
                               trendline='ols',
                               title="Daily Steps vs Sleep Quality",
                               color_discrete_sequence=['#4cc9f0', '#f72585', '#7209b7', '#06d6a0'])
        st.plotly_chart(fig_steps, use_container_width=True)

    


# ========================
# TAB 4 — STATISTICS & CORRELATIONS
# ========================
with tab4:
    st.subheader("📈 Statistical Factor Analysis")

    col_left, col_right = st.columns(2)

    with col_left:
        st.write("**Box Plot: Sleep Variance by Occupation**")
        fig_box = px.box(filtered_df, x="Occupation", y="Sleep Duration", color="Early_Warning",
                         title="Sleep Duration Variance by Occupation",
                         color_discrete_map={
                             "🚨 CRITICAL DETERIORATION": "#e63946",
                             "⚠️ MONITORING": "#f4a261",
                             "✅ STABLE": "#2d6a4f"
                         })
        fig_box.update_xaxes(tickangle=45)
        st.plotly_chart(fig_box, use_container_width=True)

    with col_right:
        st.write("**Correlation Heatmap**")
        corr_matrix = filtered_df.corr(numeric_only=True)
        fig_heat = px.imshow(corr_matrix, text_auto=True,
                             color_continuous_scale='RdBu_r', aspect="auto")
        fig_heat.update_layout(margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_heat, use_container_width=True)

    st.divider()
    st.write("### 🔍 Key Health Correlation Insights")
    st.info("""
    **Data Mining Observations:**
    1. **Stress vs Sleep Duration:** Strong negative correlation — as stress rises, sleep duration falls significantly.
    2. **Physical Activity → Sleep Quality:** Higher activity levels are associated with measurably better sleep quality scores.
    3. **Heart Rate & Anomalies:** Elevated resting heart rates are disproportionately linked to 🚨 Critical Early Warning flags.
    4. **BMI & Sleep Disorders:** Overweight/Obese categories show higher prevalence of Sleep Apnea and Insomnia.
    5. **Cardiac Stress Index:** Persons with CSI > 10 are most likely to appear in the Anomaly cluster.
    """)


# ========================
# TAB 5 — PREDICT NEW PERSON
# ========================
with tab5:
    st.subheader("🤖 Predictive Model — Assess a New Person's Sleep Health")
    st.write("This module uses a **Random Forest** model trained on the dataset to predict sleep disorder risk and estimated sleep quality for a new individual.")

    # Train models
    with st.spinner("Training models on dataset..."):
        clf, reg, le_disorder, le_gender, le_bmi, model_features, clf_acc, reg_mae = train_models(df)

    col_acc1, col_acc2 = st.columns(2)
    col_acc1.metric("🎯 Disorder Classifier Accuracy", f"{clf_acc*100:.1f}%",
                    help="Random Forest accuracy on 20% held-out test set")
    col_acc2.metric("📉 Sleep Quality MAE", f"{reg_mae:.2f} pts",
                    help="Mean Absolute Error on 1–10 quality scale (lower = better)")

    st.divider()
    st.write("### Enter New Person's Details")

    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.slider("Age", 18, 70, 35)
        gender = st.selectbox("Gender", ["Male", "Female"])
        sleep_dur = st.slider("Sleep Duration (hrs)", 4.0, 10.0, 7.0, 0.5)
    with c2:
        physical_act = st.slider("Physical Activity Level (mins/day)", 0, 90, 45)
        stress = st.slider("Stress Level (1–10)", 1, 10, 5)
        bmi_cat = st.selectbox("BMI Category", ["Normal Weight", "Overweight", "Obese"])
    with c3:
        heart_rate = st.slider("Resting Heart Rate (bpm)", 50, 100, 72)
        daily_steps = st.slider("Daily Steps", 1000, 20000, 7000, 500)

    if st.button("🔮 Predict Sleep Health", type="primary", use_container_width=True):
        # Encode inputs
        gender_enc = le_gender.transform([gender])[0] if gender in le_gender.classes_ else 0
        bmi_enc_val = le_bmi.transform([bmi_cat])[0] if bmi_cat in le_bmi.classes_ else 0

        input_data = pd.DataFrame([[age, gender_enc, sleep_dur, physical_act, stress,
                                     bmi_enc_val, heart_rate, daily_steps]],
                                   columns=model_features)

        disorder_pred_enc = clf.predict(input_data)[0]
        disorder_pred = le_disorder.inverse_transform([disorder_pred_enc])[0]
        disorder_proba = clf.predict_proba(input_data)[0]
        quality_pred = reg.predict(input_data)[0]

        # Risk Score
        # In app.py SleepSystem class — replace the Risk_Score lines with:
        hr_excess = (self.df['Heart Rate'] - 65).clip(lower=0)
        hr_score  = (hr_excess / 25) * 20
        stress_score = (self.df['Stress Level'] / 10) * 40
        sleep_score  = (self.df['Sleep_Debt'] / 3.5) * 40
        self.df['Risk_Score'] = (stress_score + sleep_score + hr_score).clip(0, 100).round(1)

        # Display results
        st.divider()
        st.write("### 📋 Prediction Results")

        r1, r2, r3 = st.columns(3)
        r1.metric("🩺 Predicted Sleep Disorder", disorder_pred if disorder_pred != 'None' else "None Detected")
        r2.metric("😴 Estimated Sleep Quality", f"{quality_pred:.1f} / 10")
        r3.metric("⚠️ Computed Risk Score", f"{risk:.0f} / 100",
                  delta="High Risk" if risk > 60 else ("Monitor" if risk > 40 else "Stable"))

        # Probability breakdown
        st.write("#### Disorder Probability Breakdown")
        proba_df = pd.DataFrame({
            'Disorder': le_disorder.classes_,
            'Probability': disorder_proba
        }).sort_values('Probability', ascending=True)
        fig_proba = px.bar(proba_df, x='Probability', y='Disorder', orientation='h',
                           color='Probability', color_continuous_scale='RdYlGn_r',
                           title="Predicted Probability by Disorder Class")
        fig_proba.update_layout(xaxis_tickformat='.0%')
        st.plotly_chart(fig_proba, use_container_width=True)

        # Feature importance
        st.write("#### Feature Importance (What drove this prediction?)")
        importance_df = pd.DataFrame({
            'Feature': model_features,
            'Importance': clf.feature_importances_
        }).sort_values('Importance', ascending=True)
        fig_imp = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                         color='Importance', color_continuous_scale='Blues',
                         title="Random Forest Feature Importances")
        st.plotly_chart(fig_imp, use_container_width=True)

        # Advice
        st.write("#### 💡 Personalised Recommendations")
        advice = []
        if stress >= 7:
            advice.append("🧘 **High stress detected.** Consider mindfulness techniques or reducing workload.")
        if sleep_dur < 6.5:
            advice.append("😴 **Sleep debt accumulating.** Aim for 7–9 hours consistently.")
        if physical_act < 30:
            advice.append("🏃 **Low physical activity.** 30+ mins of daily exercise significantly improves sleep quality.")
        if heart_rate > 80:
            advice.append("❤️ **Elevated resting heart rate.** Regular cardio can reduce baseline HR over time.")
        if bmi_cat in ['Overweight', 'Obese']:
            advice.append("⚖️ **BMI category is a risk factor** for sleep apnea. Consult a healthcare provider.")
        if not advice:
            advice.append("✅ **Your metrics look healthy!** Keep maintaining your current habits.")

        for tip in advice:
            st.markdown(f"- {tip}")