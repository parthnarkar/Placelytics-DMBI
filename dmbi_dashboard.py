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
    page_title="DMBI College Placement Analytics", 
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
st.title("Advanced DMBI College Placement Analytics Dashboard")
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
st.sidebar.title("DMBI Analytics Menu")
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
    
    high_performers = len(df_processed[df_processed['Performance_Tier'] == 'High_Performer'])
    avg_internships = df['Internships'].mean()
    avg_aptitude = df['AptitudeTestScore'].mean()
    with_training = len(df[df['PlacementTraining'] == 'Yes'])
    
    with col1:
        st.metric("High Performers", f"{high_performers}")
    with col2:
        st.metric("Avg Internships", f"{avg_internships:.1f}")
    with col3:
        st.metric("Avg Aptitude Score", f"{avg_aptitude:.1f}")
    with col4:
        st.metric("With Training", f"{with_training}")
    
    # Placement Rate by Performance Tier
    col1, col2 = st.columns(2)
    
    with col1:
        tier_analysis = df_processed.groupby('Performance_Tier')['PlacementStatus'].apply(
            lambda x: (x == 'Placed').mean() * 100
        ).reset_index()
        
        fig = px.bar(tier_analysis, x='Performance_Tier', y='PlacementStatus',
                     title="Placement Rate by Performance Tier",
                     labels={'PlacementStatus': 'Placement Rate (%)'})
        fig.update_traces(marker_color='lightblue')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        cgpa_analysis = df_processed.groupby('CGPA_Category')['PlacementStatus'].apply(
            lambda x: (x == 'Placed').mean() * 100
        ).reset_index()
        
        fig = px.pie(cgpa_analysis, values='PlacementStatus', names='CGPA_Category',
                     title="Placement Distribution by CGPA Category")
        st.plotly_chart(fig, use_container_width=True)

elif analysis_type == "Predictive Analytics":
    st.header("Predictive Analytics & Machine Learning Insights")
    
    # Enhanced Model Training with Multiple Algorithms
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
    
    # Feature Importance Analysis
    numeric_features = ['CGPA', 'Internships', 'Projects', 'Workshops/Certifications', 
                       'AptitudeTestScore', 'SoftSkillsRating', 'SSC_Marks', 'HSC_Marks',
                       'Academic_Index', 'Experience_Score', 'Competency_Score']
    
    X = df_processed[numeric_features]
    y = df_processed['PlacementStatus_encoded']
    
    # Train multiple models for ensemble prediction
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Random Forest Model
    rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Logistic Regression Model
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train, y_train)
    
    # Gradient Boosting Model
    gb_model = GradientBoostingClassifier(random_state=42, n_estimators=100)
    gb_model.fit(X_train, y_train)
    
    # Model Performance Metrics
    models = {
        'Random Forest': rf_model,
        'Logistic Regression': lr_model,
        'Gradient Boosting': gb_model
    }
    
    model_performance = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        
        model_performance[name] = {
            'accuracy': accuracy,
            'auc': auc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
    
    # Display Model Performance
    st.subheader("Model Performance Comparison")
    perf_df = pd.DataFrame(model_performance).T
    perf_df['accuracy'] = perf_df['accuracy'].apply(lambda x: f"{x:.3f}")
    perf_df['auc'] = perf_df['auc'].apply(lambda x: f"{x:.3f}")
    perf_df['cv_score'] = perf_df.apply(lambda row: f"{row['cv_mean']:.3f} ± {row['cv_std']:.3f}", axis=1)
    
    st.dataframe(perf_df[['accuracy', 'auc', 'cv_score']].rename(columns={
        'accuracy': 'Test Accuracy',
        'auc': 'AUC Score',
        'cv_score': 'CV Score (Mean ± Std)'
    }))
    
    # Feature Importance Analysis
    feature_importance = pd.DataFrame({
        'Feature': numeric_features,
        'RF_Importance': rf_model.feature_importances_,
        'LR_Coef': np.abs(lr_model.coef_[0]),
        'GB_Importance': gb_model.feature_importances_
    })
    
    # Normalize coefficients for comparison
    feature_importance['LR_Coef_Norm'] = feature_importance['LR_Coef'] / feature_importance['LR_Coef'].max()
    feature_importance['Avg_Importance'] = (feature_importance['RF_Importance'] + 
                                           feature_importance['LR_Coef_Norm'] + 
                                           feature_importance['GB_Importance']) / 3
    feature_importance = feature_importance.sort_values('Avg_Importance', ascending=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(feature_importance, x='Avg_Importance', y='Feature', orientation='h',
                     title="Average Feature Importance (Ensemble)")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Correlation Heatmap
        correlation_matrix = df_processed[numeric_features].corr()
        
        fig = px.imshow(correlation_matrix, 
                        title="Feature Correlation Heatmap",
                        color_continuous_scale="RdBu")
        st.plotly_chart(fig, use_container_width=True)
    
    # Enhanced Prediction Interface
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
    
    if st.button("Predict Placement Probability", type="primary"):
        # Calculate derived features using the same logic as training
        academic_index = cgpa * 0.4 + ssc_marks/100 * 0.3 + hsc_marks/100 * 0.3
        experience_score = internships * 2 + projects + certifications * 1.5
        competency_score = aptitude/100 * 0.6 + soft_skills/5 * 0.4
        
        # Prepare input for prediction
        new_student = np.array([[cgpa, internships, projects, certifications, aptitude,
                                soft_skills, ssc_marks, hsc_marks, academic_index,
                                experience_score, competency_score]])
        
        # Get predictions from all models
        rf_prob = rf_model.predict_proba(new_student)[0][1]
        lr_prob = lr_model.predict_proba(new_student)[0][1]
        gb_prob = gb_model.predict_proba(new_student)[0][1]
        
        # Ensemble prediction (weighted average based on model performance)
        ensemble_prob = (rf_prob * 0.4 + lr_prob * 0.35 + gb_prob * 0.25)
        
        # Display Results
        st.markdown("---")
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
            
            # Performance tier prediction
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
            # Comparison with similar students
            similar_students = df_processed[
                (abs(df_processed['CGPA'] - cgpa) <= 0.3) &
                (abs(df_processed['Internships'] - internships) <= 1) &
                (abs(df_processed['AptitudeTestScore'] - aptitude) <= 15)
            ]
        
            if len(similar_students) > 10:  # Only show if we have meaningful sample size
                similar_placement_rate = (similar_students['PlacementStatus'] == 'Placed').mean()
                st.metric("Similar Students", f"{len(similar_students)} found")
                st.metric("Their Placement Rate", f"{similar_placement_rate:.1%}")
                
                # Show prediction accuracy
                prediction_error = abs(ensemble_prob - similar_placement_rate)
                if prediction_error <= 0.15:
                    st.success("High Prediction Accuracy")
                elif prediction_error <= 0.25:
                    st.warning("Moderate Prediction Accuracy")
                else:
                    st.error("Review Prediction")
            else:
                st.write("**Unique Profile**")
                st.write("Limited similar students in dataset")
                st.write("Prediction based on model inference")        # Detailed Analysis and Recommendations
        st.subheader("Detailed Analysis & Recommendations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Strengths:**")
            strengths = []
            if cgpa >= 8.0:
                strengths.append("Excellent Academic Performance (CGPA ≥ 8.0)")
            elif cgpa >= 7.5:
                strengths.append("Good Academic Performance")
            
            if internships >= 2:
                strengths.append("Strong Internship Experience")
            elif internships >= 1:
                strengths.append("Some Internship Experience")
            
            if aptitude >= 85:
                strengths.append("Excellent Aptitude Score")
            elif aptitude >= 75:
                strengths.append("Good Aptitude Score")
            
            if soft_skills >= 4.5:
                strengths.append("Strong Soft Skills")
            
            if projects >= 3:
                strengths.append("Multiple Projects Completed")
            
            if placement_training == "Yes":
                strengths.append("Completed Placement Training")
            
            for strength in strengths:
                st.write(f"• {strength}")
        
        with col2:
            st.markdown("**Improvement Areas:**")
            improvements = []
            if cgpa < 7.5:
                improvements.append("Focus on improving CGPA")
            if internships == 0:
                improvements.append("Gain internship experience")
            if aptitude < 75:
                improvements.append("Improve aptitude test scores")
            if soft_skills < 4.0:
                improvements.append("Develop soft skills")
            if projects < 2:
                improvements.append("Work on more projects")
            if placement_training == "No":
                improvements.append("Consider placement training")
            if extracurricular == "No":
                improvements.append("Participate in extracurricular activities")
            
            for improvement in improvements:
                st.write(f"• {improvement}")
        
        # Probability Gauge Chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = ensemble_prob * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Placement Probability"},
            delta = {'reference': 42.0},  # Overall dataset placement rate
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 30], 'color': "lightgray"},
                    {'range': [30, 60], 'color': "yellow"},
                    {'range': [60, 100], 'color': "lightgreen"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90}}))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

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

# Footer
st.markdown("---")
st.markdown("""
**DMBI Concepts Implemented:**
- Data Mining: Feature Engineering, Clustering, Classification
- Business Intelligence: KPIs, Segmentation, Risk Analytics, Trend Analysis
- Advanced Analytics: Predictive Modeling, Statistical Analysis, Correlation Mining
""")
st.markdown("*Built using Advanced Data Mining & Business Intelligence Techniques*")