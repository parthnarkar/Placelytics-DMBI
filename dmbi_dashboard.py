import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Placelytics College Placement Analytics", 
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.metric-card {
    background-color: #5A67D8;
    color: white;
    padding: 1.5rem;
    border-radius: 10px;
    border-left: 5px solid #4C51BF;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    margin-bottom: 1rem;
    transition: transform 0.2s;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 12px rgba(0,0,0,0.15);
}
.metric-card h4 {
    color: #FFD700;
    font-weight: bold;
    margin-bottom: 1rem;
    font-size: 1.2rem;
}
.metric-card p {
    margin: 0.5rem 0;
    font-size: 0.95rem;
}
.insight-box {
    background-color: #2563EB;
    color: white;
    padding: 1.5rem;
    border-radius: 10px;
    border: 1px solid #1D4ED8;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}
.cluster-card-0 {
    background-color: #081e42;
    color: white;
    padding: 1.5rem;
    border-radius: 10px;
    border-left: 5px solid #c0392b;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    margin-bottom: 1rem;
    transition: transform 0.2s;
}
.cluster-card-1 {
    background-color: #081e42;
    color: white;
    padding: 1.5rem;
    border-radius: 10px;
    border-left: 5px solid #2980b9;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    margin-bottom: 1rem;
    transition: transform 0.2s;
}
.cluster-card-2 {
    background-color: #081e42;
    color: white;
    padding: 1.5rem;
    border-radius: 10px;
    border-left: 5px solid #8e44ad;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    margin-bottom: 1rem;
    transition: transform 0.2s;
}
.cluster-card-3 {
    background-color: #081e42;
    color: white;
    padding: 1.5rem;
    border-radius: 10px;
    border-left: 5px solid #229954;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    margin-bottom: 1rem;
    transition: transform 0.2s;
}
.cluster-card-0:hover, .cluster-card-1:hover, .cluster-card-2:hover, .cluster-card-3:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 16px rgba(0,0,0,0.2);
}
.cluster-card-0 h4, .cluster-card-1 h4, .cluster-card-2 h4, .cluster-card-3 h4 {
    color: #FFD700;
    font-weight: bold;
    margin-bottom: 1rem;
    font-size: 1.3rem;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
}
.cluster-card-0 p, .cluster-card-1 p, .cluster-card-2 p, .cluster-card-3 p {
    margin: 0.5rem 0;
    font-size: 1rem;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("Placelytics — Advanced College Placement Analytics Dashboard")
st.markdown("### Data Mining & Business Intelligence for Educational Success")

# Load and process data
@st.cache_data
def load_and_process_data():
    df = pd.read_csv('placementdata.csv')
    
    # Data preprocessing
    df_processed = df.copy()
    label_encoder = LabelEncoder()
    df_processed['PlacementStatus_encoded'] = label_encoder.fit_transform(df_processed['PlacementStatus'])
    df_processed['ExtracurricularActivities_encoded'] = label_encoder.fit_transform(df_processed['ExtracurricularActivities'])
    df_processed['PlacementTraining_encoded'] = label_encoder.fit_transform(df_processed['PlacementTraining'])
    
    # Feature Engineering (DMBI Concepts)
    df_processed['Academic_Index'] = (df_processed['CGPA'] * 0.4 + 
                                     df_processed['SSC_Marks']/100 * 0.3 + 
                                     df_processed['HSC_Marks']/100 * 0.3)
    
    df_processed['Experience_Score'] = (df_processed['Internships'] * 2 + 
                                       df_processed['Projects'] + 
                                       df_processed['Workshops/Certifications'] * 1.5)
    
    df_processed['Competency_Score'] = (df_processed['AptitudeTestScore']/100 * 0.6 + 
                                       df_processed['SoftSkillsRating']/5 * 0.4)
    
    # Performance Tiers (DMBI: Classification)
    conditions = [
        (df_processed['Academic_Index'] >= 8.5) & (df_processed['Experience_Score'] >= 5),
        (df_processed['Academic_Index'] >= 7.5) & (df_processed['Experience_Score'] >= 3),
        (df_processed['Academic_Index'] >= 6.5) & (df_processed['Experience_Score'] >= 1),
    ]
    choices = ['High_Performer', 'Medium_Performer', 'Low_Performer']
    df_processed['Performance_Tier'] = np.select(conditions, choices, default='Basic_Performer')
    
    # CGPA Categories (Binning)
    df_processed['CGPA_Category'] = pd.cut(df_processed['CGPA'], 
                                          bins=[0, 7.0, 8.0, 9.0, 10.0], 
                                          labels=['Below_Average', 'Average', 'Good', 'Excellent'])
    
    # Student Clustering (DMBI: Segmentation)
    cluster_features = ['Academic_Index', 'Experience_Score', 'Competency_Score']
    scaler = StandardScaler()
    cluster_data = scaler.fit_transform(df_processed[cluster_features])
    kmeans = KMeans(n_clusters=4, random_state=42)
    df_processed['Student_Cluster'] = kmeans.fit_predict(cluster_data)
    
    return df, df_processed

df, df_processed = load_and_process_data()

# Sidebar for navigation
st.sidebar.title("Placelytics Analytics Menu")
analysis_type = st.sidebar.selectbox(
    "Choose Analysis Type:",
    ["Executive Dashboard", "Predictive Analytics", "Student Segmentation", 
     "Feature Analysis", "Risk Analytics", "Trend Analysis"]
)

# Executive Dashboard
if analysis_type == "Executive Dashboard":
    st.header("Executive Dashboard - Key Performance Indicators")
    
    # KPIs Row 1
    col1, col2, col3, col4 = st.columns(4)
    
    total_students = len(df)
    placed_students = len(df[df['PlacementStatus'] == 'Placed'])
    placement_rate = placed_students / total_students
    avg_cgpa = df['CGPA'].mean()
    
    with col1:
        st.metric("Total Students", f"{total_students:,}")
    with col2:
        st.metric("Placed Students", f"{placed_students:,}")
    with col3:
        st.metric("Placement Rate", f"{placement_rate:.1%}")
    with col4:
        st.metric("Average CGPA", f"{avg_cgpa:.2f}")
    
    # KPIs Row 2
    col1, col2, col3, col4 = st.columns(4)
    
    # Replace previous 'High Performers' metric with a cluster-based high achiever count
    high_achiever_cluster_count = int(df_processed[df_processed['Student_Cluster'] == 3].shape[0])
    avg_internships = df['Internships'].mean()
    avg_aptitude = df['AptitudeTestScore'].mean()
    with_training = len(df[df['PlacementTraining'] == 'Yes'])
    
    with col1:
        st.metric("High Achiever Cluster (Count)", f"{high_achiever_cluster_count}")
    with col2:
        st.metric("Avg Internships", f"{avg_internships:.1f}")
    with col3:
        st.metric("Avg Aptitude Score", f"{avg_aptitude:.1f}")
    with col4:
        st.metric("With Training", f"{with_training}")
    
    # Spacer to increase bottom padding under KPIs so charts appear lower on the page
    st.markdown("<div style='height:48px;'></div>", unsafe_allow_html=True)
    # Placement Rate by Performance Tier
    # Create 5 columns: left, center-left (pie), gap, center-right (stack), right
    # make center columns wider (center-left and center-right bigger) and keep a small gap
    col_left, col_center_left, gap_col, col_center_right, col_right = st.columns([0.8,2,0.4,2,0.8])

    with col_left:
        # Stacked bar: counts of Placed vs NotPlaced per Performance_Tier
        expected_tiers = ['Basic_Performer']
        placed_counts = df_processed[df_processed['PlacementStatus'] == 'Placed']['Performance_Tier'].value_counts().reindex(expected_tiers).fillna(0)
        notplaced_counts = df_processed[df_processed['PlacementStatus'] != 'Placed']['Performance_Tier'].value_counts().reindex(expected_tiers).fillna(0)
        total_counts = placed_counts + notplaced_counts
        placement_rates = (placed_counts / total_counts * 100).fillna(0)

        # Create horizontal stacked bars (tiers on y-axis)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=expected_tiers,
            x=placed_counts.values,
            orientation='h',
            name='Placed',
            text=placed_counts.values,            # show counts inside bars
            textposition='inside',
            width=0.4,                             # reduce bar thickness
            marker_color='#10B981',
            hovertemplate='%{y}<br>Placed: %{x}<extra></extra>'
        ))
        fig.add_trace(go.Bar(
            y=expected_tiers,
            x=notplaced_counts.values,
            orientation='h',
            name='Not Placed',
            text=notplaced_counts.values,        # show counts inside bars
            textposition='inside',
            width=0.4,                             # reduce bar thickness
            marker_color='#EF4444',
            hovertemplate='%{y}<br>Not Placed: %{x}<extra></extra>'
        ))

        # Add placement rate annotations to the right of each stacked bar
        annotations = []
        max_total = max(total_counts.values) if len(total_counts.values) > 0 else 0
        for i, tier in enumerate(expected_tiers):
            annotations.append(dict(
                x=total_counts.values[i] + max_total*0.02,
                y=tier,
                text=f"{placement_rates.values[i]:.1f}%",
                showarrow=False,
                xanchor='left',
                font=dict(color='white', size=12)
            ))

    fig.update_layout(
        barmode='stack',
        title='Placed vs Not Placed by Performance Tier (with Placement %)',
        template='plotly_dark',
        annotations=annotations,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        bargap=0.4,
    )

    fig.update_xaxes(title_text='Number of Students')
    fig.update_yaxes(title_text='Performance Tier', automargin=True)
    st.plotly_chart(fig, use_container_width=True)

    with col_center_left:  # left center: CGPA pie
        cgpa_analysis = df_processed.groupby('CGPA_Category')['PlacementStatus'].apply(
            lambda x: (x == 'Placed').mean() * 100
        ).reset_index()

        fig = px.pie(cgpa_analysis, values='PlacementStatus', names='CGPA_Category',
                     title="Placement Distribution by CGPA Category")
        fig.update_traces(textinfo='percent+label')
        fig.update_layout(margin=dict(t=40, b=20, l=20, r=20), height=420)
        st.plotly_chart(fig, use_container_width=True)

    with col_center_right:  # right center: stacked counts by CGPA category
        cgpa_counts = df_processed.groupby(['CGPA_Category', 'PlacementStatus']).size().unstack(fill_value=0)
        # Ensure consistent ordering of categories
        category_order = ['Below_Average', 'Average', 'Good', 'Excellent']
        cgpa_counts = cgpa_counts.reindex(category_order).fillna(0)

        fig2 = go.Figure()
        if 'Placed' in cgpa_counts.columns:
            fig2.add_trace(go.Bar(name='Placed', x=cgpa_counts.index.astype(str), y=cgpa_counts['Placed'], marker_color='#10B981'))
        if 'NotPlaced' in cgpa_counts.columns:
            # handle both 'NotPlaced' and 'Not Placed' keys
            y_vals = cgpa_counts.get('NotPlaced') if 'NotPlaced' in cgpa_counts.columns else cgpa_counts.get('Not Placed')
            if y_vals is not None:
                fig2.add_trace(go.Bar(name='Not Placed', x=cgpa_counts.index.astype(str), y=y_vals, marker_color='#EF4444'))

        fig2.update_layout(barmode='stack', title='Counts by CGPA Category (Placed vs Not Placed)', template='plotly_dark', margin=dict(t=40, b=20), height=420)
        fig2.update_xaxes(title_text='CGPA Category')
        fig2.update_yaxes(title_text='Number of Students')
        st.plotly_chart(fig2, use_container_width=True)

elif analysis_type == "Predictive Analytics":
    st.header("Predictive Analytics & Machine Learning Insights")
    st.info("Models are trained only when you click 'Predict Placement Probability' to avoid repeated long loading during UI interaction.")

    # Feature list for modeling
    numeric_features = ['CGPA', 'Internships', 'Projects', 'Workshops/Certifications', 
                       'AptitudeTestScore', 'SoftSkillsRating', 'SSC_Marks', 'HSC_Marks',
                       'Academic_Index', 'Experience_Score', 'Competency_Score']

    # Helper: train models (run only on demand)
    def train_models(df_proc, numeric_features):
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
        from sklearn.metrics import accuracy_score, roc_auc_score

        X = df_proc[numeric_features]
        y = df_proc['PlacementStatus_encoded']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
        rf_model.fit(X_train, y_train)

        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        lr_model.fit(X_train, y_train)

        gb_model = GradientBoostingClassifier(random_state=42, n_estimators=100)
        gb_model.fit(X_train, y_train)

        models = {
            'Random Forest': rf_model,
            'Logistic Regression': lr_model,
            'Gradient Boosting': gb_model
        }

        model_performance = {}
        for name, model in models.items():
            y_pred = model.predict(X_test)
            try:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            except Exception:
                y_pred_proba = np.zeros(len(y_test))
            accuracy = accuracy_score(y_test, y_pred)
            try:
                auc = roc_auc_score(y_test, y_pred_proba)
            except Exception:
                auc = 0.0
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)

            model_performance[name] = {
                'accuracy': accuracy,
                'auc': auc,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }

        perf_df = pd.DataFrame(model_performance).T

        feature_importance = pd.DataFrame({
            'Feature': numeric_features,
            'RF_Importance': rf_model.feature_importances_,
            'LR_Coef': np.abs(lr_model.coef_[0]),
            'GB_Importance': gb_model.feature_importances_
        })
        feature_importance['LR_Coef_Norm'] = feature_importance['LR_Coef'] / feature_importance['LR_Coef'].max()
        feature_importance['Avg_Importance'] = (feature_importance['RF_Importance'] + 
                                               feature_importance['LR_Coef_Norm'] + 
                                               feature_importance['GB_Importance']) / 3

        return models, perf_df, feature_importance

    # Enhanced Prediction Interface (inputs only)
    st.subheader("Individual Student Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Academic Information**")
        cgpa = st.slider("CGPA", 6.0, 10.0, 7.5, 0.1)
        ssc_marks = st.slider("SSC Marks (%)", 50, 100, 75)
        hsc_marks = st.slider("HSC Marks (%)", 50, 100, 75)
        aptitude = st.slider("Aptitude Test Score", 50, 100, 75)

    with col2:
        st.markdown("**Experience & Skills**")
        internships = st.selectbox("Number of Internships", [0, 1, 2, 3, 4])
        projects = st.selectbox("Number of Projects", [0, 1, 2, 3, 4, 5])
        certifications = st.selectbox("Workshops/Certifications", [0, 1, 2, 3, 4, 5])
        soft_skills = st.slider("Soft Skills Rating", 1.0, 5.0, 4.0, 0.1)

    with col3:
        st.markdown("**Additional Factors**")
        extracurricular = st.selectbox("Extracurricular Activities", ["No", "Yes"])
        placement_training = st.selectbox("Placement Training", ["No", "Yes"])

        # Additional info display
        st.markdown("**Quick Stats**")
        st.write(f"Dataset Average CGPA: {df['CGPA'].mean():.2f}")
        st.write(f"Dataset Placement Rate: {(df['PlacementStatus'] == 'Placed').mean():.1%}")

    # Only train models and run predictions when user explicitly requests it
    if st.button("Predict Placement Probability", type="primary"):
        with st.spinner("Training models and computing predictions (this may take a moment)..."):
            if not st.session_state.get('models_trained', False):
                models_dict, perf_df, feature_importance = train_models(df_processed, numeric_features)
                st.session_state['models_dict'] = models_dict
                st.session_state['perf_df'] = perf_df
                st.session_state['feature_importance'] = feature_importance
                st.session_state['models_trained'] = True
            else:
                models_dict = st.session_state['models_dict']
                perf_df = st.session_state['perf_df']
                feature_importance = st.session_state['feature_importance']

            # Prepare derived features and input
            academic_index = cgpa * 0.4 + ssc_marks/100 * 0.3 + hsc_marks/100 * 0.3
            experience_score = internships * 2 + projects + certifications * 1.5
            competency_score = aptitude/100 * 0.6 + soft_skills/5 * 0.4

            new_student = np.array([[cgpa, internships, projects, certifications, aptitude,
                                    soft_skills, ssc_marks, hsc_marks, academic_index,
                                    experience_score, competency_score]])

            rf_model = models_dict['Random Forest']
            lr_model = models_dict['Logistic Regression']
            gb_model = models_dict['Gradient Boosting']

            rf_prob = rf_model.predict_proba(new_student)[0][1]
            lr_prob = lr_model.predict_proba(new_student)[0][1]
            gb_prob = gb_model.predict_proba(new_student)[0][1]

            ensemble_prob = (rf_prob * 0.4 + lr_prob * 0.35 + gb_prob * 0.25)

        # After training & prediction, show performance and feature importance
        st.markdown("---")
        st.subheader("Model Performance Comparison")
        perf_df_display = perf_df.copy()
        perf_df_display['accuracy'] = perf_df_display['accuracy'].apply(lambda x: f"{x:.3f}")
        perf_df_display['auc'] = perf_df_display['auc'].apply(lambda x: f"{x:.3f}")
        perf_df_display['cv_score'] = perf_df_display.apply(lambda row: f"{row['cv_mean']:.3f} ± {row['cv_std']:.3f}", axis=1)
        st.dataframe(perf_df_display[['accuracy', 'auc', 'cv_score']].rename(columns={
            'accuracy': 'Test Accuracy', 'auc': 'AUC Score', 'cv_score': 'CV Score (Mean ± Std)'
        }))

        # Feature importance visuals
        feature_importance = st.session_state['feature_importance']
        feature_importance = feature_importance.sort_values('Avg_Importance', ascending=True)
        colA, colB = st.columns(2)
        with colA:
            fig = px.bar(feature_importance, x='Avg_Importance', y='Feature', orientation='h', title="Average Feature Importance (Ensemble)")
            st.plotly_chart(fig, use_container_width=True)
        with colB:
            correlation_matrix = df_processed[numeric_features].corr()
            fig = px.imshow(correlation_matrix, title="Feature Correlation Heatmap", color_continuous_scale="RdBu")
            st.plotly_chart(fig, use_container_width=True)

        # Display prediction results as before
        st.subheader("Prediction Results")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if ensemble_prob > 0.75:
                st.success("VERY HIGH PROBABILITY")
                confidence = "Very High"
                color = "green"
            elif ensemble_prob > 0.6:
                st.success("HIGH PROBABILITY")
                confidence = "High"
                color = "lightgreen"
            elif ensemble_prob > 0.45:
                st.warning("MODERATE PROBABILITY")
                confidence = "Moderate"
                color = "orange"
            elif ensemble_prob > 0.3:
                st.warning("LOW PROBABILITY")
                confidence = "Low"
                color = "orange"
            else:
                st.error("VERY LOW PROBABILITY")
                confidence = "Very Low"
                color = "red"
            st.metric("Final Prediction", f"{ensemble_prob:.1%}")
            st.metric("Confidence Level", confidence)

        with col2:
            st.metric("Academic Index", f"{academic_index:.2f}")
            st.metric("Experience Score", f"{experience_score:.1f}")
            st.metric("Competency Score", f"{competency_score:.2f}")

        with col3:
            st.write("**Individual Model Predictions:**")
            st.write(f"Random Forest: {rf_prob:.1%}")
            st.write(f"Logistic Regression: {lr_prob:.1%}")
            st.write(f"Gradient Boosting: {gb_prob:.1%}")
            if academic_index >= 3.8 and experience_score >= 5:
                tier = "High Performer"
            elif academic_index >= 3.5 and experience_score >= 3:
                tier = "Medium Performer"
            elif academic_index >= 3.2 and experience_score >= 1:
                tier = "Developing Performer"
            else:
                tier = "Needs Improvement"
            st.write(f"**Performance Tier:** {tier}")

        with col4:
            similar_students = df_processed[
                (abs(df_processed['CGPA'] - cgpa) <= 0.3) &
                (abs(df_processed['Internships'] - internships) <= 1) &
                (abs(df_processed['AptitudeTestScore'] - aptitude) <= 15)
            ]
            if len(similar_students) > 10:
                similar_placement_rate = (similar_students['PlacementStatus'] == 'Placed').mean()
                st.metric("Similar Students", f"{len(similar_students)} found")
                st.metric("Their Placement Rate", f"{similar_placement_rate:.1%}")
            else:
                st.write("**Unique Profile**")
                st.write("Limited similar students in dataset")
                st.write("Prediction based on model inference")

elif analysis_type == "Student Segmentation":
    st.header("Student Segmentation Analysis (Clustering)")
    
    # Cluster Analysis
    cluster_analysis = df_processed.groupby('Student_Cluster').agg({
        'Academic_Index': 'mean',
        'Experience_Score': 'mean',
        'Competency_Score': 'mean',
        'PlacementStatus': lambda x: (x == 'Placed').mean()
    }).round(3)
    
    cluster_analysis['Student_Count'] = df_processed.groupby('Student_Cluster').size()
    cluster_analysis = cluster_analysis.reset_index()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 3D Scatter Plot
        fig = px.scatter_3d(df_processed, x='Academic_Index', y='Experience_Score', z='Competency_Score',
                           color='Student_Cluster', title="3D Student Segmentation")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Cluster Characteristics
        st.subheader("Cluster Characteristics")
        
        cluster_names = {
            0: "Moderate Performers",
            1: "Developing Students", 
            2: "Average Achievers",
            3: "High Achievers"
        }
        
        cluster_descriptions = {
            0: "Students with balanced academic and experience profiles",
            1: "Students who need focused development support",
            2: "Students with average performance across metrics", 
            3: "Top-performing students with excellent outcomes"
        }
        
        for _, row in cluster_analysis.iterrows():
            cluster_id = int(row['Student_Cluster'])
            placement_rate = row['PlacementStatus']
            
            # Determine performance level for better messaging
            if placement_rate >= 0.7:
                performance_text = "Excellent"
            elif placement_rate >= 0.4:
                performance_text = "Good"
            elif placement_rate >= 0.2:
                performance_text = "Needs Attention"
            else:
                performance_text = "High Risk"
            
            st.markdown(f"""
            <div class="cluster-card-{cluster_id}">
                <h4>{cluster_names.get(cluster_id, f'Cluster {cluster_id}')}</h4>
                <p><strong>Total Students:</strong> {row['Student_Count']:,}</p>
                <p><strong>Placement Rate:</strong> {placement_rate:.1%}</p>
                <p><strong>Academic Index:</strong> {row['Academic_Index']:.2f}/4.0</p>
                <p><strong>Experience Score:</strong> {row['Experience_Score']:.2f}</p>
                <p><strong>Competency Score:</strong> {row['Competency_Score']:.2f}</p>
                <p><strong>Performance Level:</strong> {performance_text}</p>
                <p style="font-style: italic; margin-top: 1rem; opacity: 0.9;">{cluster_descriptions.get(cluster_id, '')}</p>
            </div>
            """, unsafe_allow_html=True)

elif analysis_type == "Feature Analysis":
    st.header("Advanced Feature Analysis")
    
    # Feature distribution analysis
    feature = st.selectbox("Select Feature for Analysis:", 
                          ['CGPA', 'Internships', 'AptitudeTestScore', 'SoftSkillsRating'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribution by placement status
        fig = px.histogram(df, x=feature, color='PlacementStatus', 
                          title=f"{feature} Distribution by Placement Status",
                          barmode='overlay', opacity=0.7)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Box plot
        fig = px.box(df, x='PlacementStatus', y=feature,
                     title=f"{feature} Box Plot by Placement Status")
        st.plotly_chart(fig, use_container_width=True)
    
    # Statistical insights
    placed_mean = df[df['PlacementStatus'] == 'Placed'][feature].mean()
    not_placed_mean = df[df['PlacementStatus'] == 'NotPlaced'][feature].mean()
    difference = placed_mean - not_placed_mean
    
    st.markdown(f"""
    <div class="insight-box">
        <h4>Statistical Insights for {feature}</h4>
        <p><strong>Placed Students Average:</strong> {placed_mean:.2f}</p>
        <p><strong>Not Placed Students Average:</strong> {not_placed_mean:.2f}</p>
        <p><strong>Difference:</strong> {difference:.2f}</p>
        <p><strong>Impact:</strong> {'Positive' if difference > 0 else 'Negative'} correlation with placement</p>
    </div>
    """, unsafe_allow_html=True)

elif analysis_type == "Risk Analytics":
    st.header("Risk Analytics - At-Risk Student Identification")
    
    # Define risk criteria
    risk_factors = {
        'Low CGPA': df_processed['CGPA'] < 7.5,
        'No Internships': df_processed['Internships'] == 0,
        'Low Aptitude': df_processed['AptitudeTestScore'] < 70,
        'No Training': df_processed['PlacementTraining'] == 'No',
        'Low Experience': df_processed['Experience_Score'] < 2
    }
    
    # Calculate risk scores
    df_processed['Risk_Score'] = sum(risk_factors.values())
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk distribution
        risk_dist = df_processed['Risk_Score'].value_counts().sort_index()
        fig = px.bar(x=risk_dist.index, y=risk_dist.values,
                     title="Risk Score Distribution",
                     labels={'x': 'Risk Score', 'y': 'Number of Students'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Placement rate by risk score
        risk_placement = df_processed.groupby('Risk_Score')['PlacementStatus'].apply(
            lambda x: (x == 'Placed').mean() * 100
        ).reset_index()
        
        fig = px.line(risk_placement, x='Risk_Score', y='PlacementStatus',
                      title="Placement Rate vs Risk Score",
                      labels={'PlacementStatus': 'Placement Rate (%)'})
        st.plotly_chart(fig, use_container_width=True)
    
    # High-risk students
    high_risk = df_processed[df_processed['Risk_Score'] >= 3]
    st.subheader(f"High-Risk Students (Risk Score ≥ 3): {len(high_risk)} students")
    
    if len(high_risk) > 0:
        st.write("Top risk factors among high-risk students:")
        for factor, condition in risk_factors.items():
            count = sum(condition & (df_processed['Risk_Score'] >= 3))
            percentage = count / len(high_risk) * 100
            st.write(f"- {factor}: {count} students ({percentage:.1f}%)")

elif analysis_type == "Trend Analysis":
    st.header("Trend Analysis & Business Intelligence")
    
    # Performance trends
    col1, col2 = st.columns(2)
    
    with col1:
        # CGPA trends
        cgpa_bins = pd.cut(df['CGPA'], bins=5)
        cgpa_trends = df.groupby(cgpa_bins)['PlacementStatus'].apply(
            lambda x: (x == 'Placed').mean() * 100
        ).reset_index()
        cgpa_trends['CGPA_Range'] = cgpa_trends['CGPA'].astype(str)
        
        fig = px.line(cgpa_trends, x='CGPA_Range', y='PlacementStatus',
                      title="Placement Rate Trend by CGPA Range",
                      labels={'PlacementStatus': 'Placement Rate (%)'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Experience vs placement
        exp_placement = df.groupby('Internships')['PlacementStatus'].apply(
            lambda x: (x == 'Placed').mean() * 100
        ).reset_index()
        
        fig = px.bar(exp_placement, x='Internships', y='PlacementStatus',
                     title="Placement Rate by Number of Internships",
                     labels={'PlacementStatus': 'Placement Rate (%)'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Key insights
    st.subheader("Key Business Intelligence Insights")
    
    insights = [
        f"Students with CGPA ≥ 8.0 have {(df[df['CGPA'] >= 8.0]['PlacementStatus'] == 'Placed').mean():.1%} placement rate",
        f"Students with ≥ 2 internships have {(df[df['Internships'] >= 2]['PlacementStatus'] == 'Placed').mean():.1%} placement rate",
        f"Students with training have {(df[df['PlacementTraining'] == 'Yes']['PlacementStatus'] == 'Placed').mean():.1%} placement rate",
        f"High performers achieve {(df_processed[df_processed['Performance_Tier'] == 'High_Performer']['PlacementStatus'] == 'Placed').mean():.1%} placement rate"
    ]
    
    for insight in insights:
        st.markdown(f"- {insight}")

# Dynamic footers per analysis type
footer_texts = {
        "Executive Dashboard": """
        <div>
            <h4 style='margin:0 0 0.5rem 0;'>Executive Dashboard Summary</h4>
            <ul>
                <li>High-level KPIs and placement performance snapshots</li>
                <li>Placement Rate comparisons by performance tiers and CGPA</li>
                <li>Actionable items: focus areas for improving institutional placement outcomes</li>
            </ul>
        </div>
        """,
        "Predictive Analytics": """
        <div>
            <h4 style='margin:0 0 0.5rem 0;'>Predictive Analytics Notes</h4>
            <ul>
                <li>Models trained in-session (Random Forest, Logistic Regression, Gradient Boosting)</li>
                <li>Use the Individual Student Prediction panel to estimate placement probability</li>
                <li>Interpret probabilities with caution and combine with domain knowledge for decisions</li>
            </ul>
        </div>
        """,
        "Student Segmentation": """
        <div>
            <h4 style='margin:0 0 0.5rem 0;'>Student Segmentation Insights</h4>
            <ul>
                <li>Students are clustered by Academic Index, Experience Score and Competency Score</li>
                <li>Use cluster characteristics to design targeted interventions and programs</li>
                <li>Monitor cluster placement rates to evaluate program effectiveness</li>
            </ul>
        </div>
        """,
        "Feature Analysis": """
        <div>
            <h4 style='margin:0 0 0.5rem 0;'>Feature Analysis Guidance</h4>
            <ul>
                <li>Explore feature distributions and correlations to identify strong predictors</li>
                <li>Leverage insights to prioritize student support and curriculum adjustments</li>
                <li>Statistical summaries are provided for quick comparisons</li>
            </ul>
        </div>
        """,
        "Risk Analytics": """
        <div>
            <h4 style='margin:0 0 0.5rem 0;'>Risk Analytics & Intervention</h4>
            <ul>
                <li>Identifies at-risk students using combined criteria (CGPA, internships, aptitude, training)</li>
                <li>Provides counts and major risk drivers to help prioritize interventions</li>
                <li>Use this view to plan targeted mentoring and upskilling programs</li>
            </ul>
        </div>
        """,
        "Trend Analysis": """
        <div>
            <h4 style='margin:0 0 0.5rem 0;'>Trend Analysis Summary</h4>
            <ul>
                <li>Displays placement trends across CGPA ranges and experience levels</li>
                <li>Helps evaluate long-term program impact and placement strategies</li>
                <li>Use trends to inform policy and training investments</li>
            </ul>
        </div>
        """
}

default_footer = """
<div>
    <h4 style='margin:0 0 0.5rem 0;'>Placelytics Concepts Implemented</h4>
    <ul>
        <li>Data Mining: Feature Engineering, Clustering, Classification</li>
        <li>Business Intelligence: KPIs, Segmentation, Risk Analytics, Trend Analysis</li>
        <li>Advanced Analytics: Predictive Modeling, Statistical Analysis, Correlation Mining</li>
    </ul>
</div>
"""

# Render the footer specific to the active analysis type
st.markdown("---")
st.markdown(footer_texts.get(analysis_type, default_footer), unsafe_allow_html=True)
st.markdown(
    """
    <h4 style="text-align:center; margin-top:5rem;">
        Built with ❤️ by 
        <a href="https://www.instagram.com/parth.builds" target="_blank" style="text-decoration:none; color:#FF4B4B;">
            @parth.builds
        </a>
    </h4>
    """,
    unsafe_allow_html=True
)
