# 🚀 College Placement Analysis with Advanced DMBI Techniques
# Enhanced with Data Mining and Business Intelligence Concepts

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2, RFE
from scipy import stats
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings('ignore')

print("🔬 ADVANCED DATA MINING & BUSINESS INTELLIGENCE ANALYSIS")
print("🎯 College Placement Prediction with DMBI Concepts")
print("="*80)

# ===================================================================
# 1. DATA PREPROCESSING & EXPLORATION (DMBI Concept: Data Preparation)
# ===================================================================

print("\n📊 Phase 1: Data Mining - Data Preparation & Exploration")
print("-" * 60)

# Load dataset
df = pd.read_csv('../../data/placementdata.csv')
print(f"✅ Dataset loaded: {df.shape[0]} records, {df.shape[1]} features")

# Data Quality Assessment (DMBI: Data Quality)
print(f"\n🔍 Data Quality Assessment:")
print(f"Missing values: {df.isnull().sum().sum()}")
print(f"Duplicate records: {df.duplicated().sum()}")
print(f"Data types: {df.dtypes.value_counts().to_dict()}")

# Statistical Summary (DMBI: Descriptive Analytics)
print(f"\n📈 Descriptive Analytics:")
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if col != 'StudentID':
        print(f"{col}: Mean={df[col].mean():.2f}, Std={df[col].std():.2f}, "
              f"Skew={df[col].skew():.2f}")

# ===================================================================
# 2. ADVANCED DATA PREPROCESSING (DMBI: Feature Engineering)
# ===================================================================

print(f"\n🔧 Phase 2: Feature Engineering & Data Transformation")
print("-" * 60)

df_processed = df.copy()

# Encode categorical variables
label_encoder = LabelEncoder()
df_processed['PlacementStatus_encoded'] = label_encoder.fit_transform(df_processed['PlacementStatus'])
df_processed['ExtracurricularActivities_encoded'] = label_encoder.fit_transform(df_processed['ExtracurricularActivities'])
df_processed['PlacementTraining_encoded'] = label_encoder.fit_transform(df_processed['PlacementTraining'])

# Feature Engineering (DMBI: Derived Attributes)
print("🛠️ Creating Derived Features:")

# 1. Academic Performance Index
df_processed['Academic_Index'] = (df_processed['CGPA'] * 0.4 + 
                                 df_processed['SSC_Marks']/100 * 0.3 + 
                                 df_processed['HSC_Marks']/100 * 0.3)

# 2. Experience Score
df_processed['Experience_Score'] = (df_processed['Internships'] * 2 + 
                                   df_processed['Projects'] + 
                                   df_processed['Workshops/Certifications'] * 1.5)

# 3. Overall Competency Score
df_processed['Competency_Score'] = (df_processed['AptitudeTestScore']/100 * 0.6 + 
                                   df_processed['SoftSkillsRating']/5 * 0.4)

# 4. Activity Participation Score
df_processed['Activity_Score'] = (df_processed['ExtracurricularActivities_encoded'] + 
                                 df_processed['PlacementTraining_encoded'])

# 5. CGPA Categories (Binning - DMBI Concept)
df_processed['CGPA_Category'] = pd.cut(df_processed['CGPA'], 
                                      bins=[0, 7.0, 8.0, 9.0, 10.0], 
                                      labels=['Below_Average', 'Average', 'Good', 'Excellent'])

# Performance Tier (DMBI: Classification)
conditions = [
    (df_processed['Academic_Index'] >= 8.5) & (df_processed['Experience_Score'] >= 5),
    (df_processed['Academic_Index'] >= 7.5) & (df_processed['Experience_Score'] >= 3),
    (df_processed['Academic_Index'] >= 6.5) & (df_processed['Experience_Score'] >= 1),
]
choices = ['High_Performer', 'Medium_Performer', 'Low_Performer']
df_processed['Performance_Tier'] = np.select(conditions, choices, default='Basic_Performer')

# Fix the performance tier logic
df_processed.loc[df_processed['Academic_Index'] >= 8.0, 'Performance_Tier'] = 'High_Performer'
df_processed.loc[(df_processed['Academic_Index'] >= 7.0) & (df_processed['Academic_Index'] < 8.0), 'Performance_Tier'] = 'Medium_Performer'
df_processed.loc[df_processed['Academic_Index'] < 7.0, 'Performance_Tier'] = 'Low_Performer'

print(f"✅ Created 6 derived features")
print(f"📊 Performance Tier Distribution:")
print(df_processed['Performance_Tier'].value_counts())

# ===================================================================
# 3. STATISTICAL ANALYSIS (DMBI: Statistical Methods)
# ===================================================================

print(f"\n📊 Phase 3: Statistical Analysis & Hypothesis Testing")
print("-" * 60)

# Chi-Square Test (DMBI: Association Analysis)
print("🔬 Chi-Square Tests for Feature Association:")

categorical_features = ['ExtracurricularActivities', 'PlacementTraining', 'Performance_Tier']
for feature in categorical_features:
    contingency_table = pd.crosstab(df_processed[feature], df_processed['PlacementStatus'])
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    print(f"{feature}: Chi2={chi2:.3f}, p-value={p_value:.6f}, "
          f"Association={'Strong' if p_value < 0.001 else 'Moderate' if p_value < 0.05 else 'Weak'}")

# Correlation Analysis (DMBI: Correlation Mining)
print(f"\n🔗 Correlation Analysis:")
numeric_features = ['CGPA', 'Internships', 'Projects', 'Workshops/Certifications', 
                   'AptitudeTestScore', 'SoftSkillsRating', 'SSC_Marks', 'HSC_Marks',
                   'Academic_Index', 'Experience_Score', 'Competency_Score']

correlation_matrix = df_processed[numeric_features + ['PlacementStatus_encoded']].corr()
placement_correlations = correlation_matrix['PlacementStatus_encoded'].sort_values(ascending=False)

print("Top 5 Features Correlated with Placement:")
placement_items = list(placement_correlations.head(6).items())[1:]
for i, (feature, corr) in enumerate(placement_items, 1):
    print(f"{i}. {feature}: {corr:.3f}")

# ===================================================================
# 4. CLUSTERING ANALYSIS (DMBI: Unsupervised Learning)
# ===================================================================

print(f"\n🎯 Phase 4: Clustering Analysis (Student Segmentation)")
print("-" * 60)

# Prepare data for clustering
cluster_features = ['Academic_Index', 'Experience_Score', 'Competency_Score', 'Activity_Score']
scaler = StandardScaler()
cluster_data = scaler.fit_transform(df_processed[cluster_features])

# K-Means Clustering (DMBI: Customer Segmentation equivalent)
print("🔍 K-Means Clustering for Student Segmentation:")

# Find optimal number of clusters using Elbow Method
inertias = []
K_range = range(2, 8)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(cluster_data)
    inertias.append(kmeans.inertia_)

# Use 4 clusters for student segmentation
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df_processed['Student_Cluster'] = kmeans.fit_predict(cluster_data)

# Analyze clusters
print(f"📊 Student Cluster Analysis (K={optimal_k}):")
for cluster_id in range(optimal_k):
    cluster_data_analysis = df_processed[df_processed['Student_Cluster'] == cluster_id]
    placement_rate = (cluster_data_analysis['PlacementStatus'] == 'Placed').mean()
    avg_academic = cluster_data_analysis['Academic_Index'].mean()
    avg_experience = cluster_data_analysis['Experience_Score'].mean()
    
    print(f"Cluster {cluster_id}: {len(cluster_data_analysis)} students, "
          f"Placement Rate: {placement_rate:.1%}, "
          f"Avg Academic: {avg_academic:.2f}, "
          f"Avg Experience: {avg_experience:.2f}")

# ===================================================================
# 5. DIMENSIONALITY REDUCTION (DMBI: PCA)
# ===================================================================

print(f"\n📐 Phase 5: Dimensionality Reduction & Feature Selection")
print("-" * 60)

# Principal Component Analysis
X_features = df_processed[numeric_features]
X_scaled = StandardScaler().fit_transform(X_features)

pca = PCA()
pca.fit(X_scaled)

# Explained variance ratio
print("🔬 PCA Analysis:")
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
print(f"Components needed for 95% variance: {n_components_95}")
print(f"First 3 components explain: {cumulative_variance[2]:.1%} of variance")

# Feature Selection using SelectKBest (DMBI: Feature Selection)
X = df_processed[numeric_features]
y = df_processed['PlacementStatus_encoded']

# Use f_classif for continuous features with binary target
from sklearn.feature_selection import f_classif
selector = SelectKBest(score_func=f_classif, k=8)
X_selected = selector.fit_transform(X, y)
selected_features = np.array(numeric_features)[selector.get_support()]

print(f"🎯 Top 8 Selected Features by F-Score:")
for i, feature in enumerate(selected_features, 1):
    print(f"{i}. {feature}")

# ===================================================================
# 6. ADVANCED MACHINE LEARNING (DMBI: Predictive Analytics)
# ===================================================================

print(f"\n🤖 Phase 6: Advanced Predictive Analytics")
print("-" * 60)

# Prepare data for ML
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature scaling for selected algorithms
scaler_ml = StandardScaler()
X_train_scaled = scaler_ml.fit_transform(X_train)
X_test_scaled = scaler_ml.transform(X_test)

# Advanced ML Models (DMBI: Ensemble Methods)
advanced_models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Support Vector Machine': SVC(probability=True, random_state=42),
    'Naive Bayes': GaussianNB(),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
}

# Model evaluation with cross-validation (DMBI: Model Validation)
print("🔍 Advanced Model Evaluation with Cross-Validation:")
results_advanced = []

for model_name, model in advanced_models.items():
    # Cross-validation
    if model_name in ['Support Vector Machine', 'K-Nearest Neighbors', 'Logistic Regression']:
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    results_advanced.append({
        'Model': model_name,
        'CV_Mean': cv_scores.mean(),
        'CV_Std': cv_scores.std(),
        'Test_Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1_Score': f1,
        'AUC': auc
    })
    
    print(f"{model_name}:")
    print(f"  CV Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    print(f"  Test Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")

# ===================================================================
# 7. BUSINESS INTELLIGENCE DASHBOARD METRICS
# ===================================================================

print(f"\n📊 Phase 7: Business Intelligence Metrics & KPIs")
print("-" * 60)

# Key Performance Indicators (KPIs)
total_students = len(df)
placed_students = len(df[df['PlacementStatus'] == 'Placed'])
placement_rate = placed_students / total_students

print("🎯 Key Performance Indicators (KPIs):")
print(f"Total Students: {total_students:,}")
print(f"Placed Students: {placed_students:,}")
print(f"Overall Placement Rate: {placement_rate:.1%}")

# Segmentation Analysis (DMBI: Business Segmentation)
print(f"\n📈 Performance Tier Analysis:")
tier_analysis = df_processed.groupby('Performance_Tier').agg({
    'PlacementStatus': lambda x: (x == 'Placed').mean(),
    'CGPA': 'mean',
    'Academic_Index': 'mean',
    'Experience_Score': 'mean'
}).round(3)

for tier, data in tier_analysis.iterrows():
    print(f"{tier}: {data['PlacementStatus']:.1%} placement rate, "
          f"Avg CGPA: {data['CGPA']:.2f}")

# Risk Analysis (DMBI: Risk Analytics)
print(f"\n⚠️ Risk Analysis - Students at Risk of Not Being Placed:")
at_risk_criteria = (
    (df_processed['CGPA'] < 7.5) |
    (df_processed['Internships'] == 0) |
    (df_processed['AptitudeTestScore'] < 70)
)
at_risk_students = df_processed[at_risk_criteria]
print(f"Students at Risk: {len(at_risk_students)} ({len(at_risk_students)/len(df):.1%})")

# Trend Analysis (DMBI: Trend Analytics)
print(f"\n📊 Trend Analysis by CGPA Categories:")
cgpa_trends = df_processed.groupby('CGPA_Category')['PlacementStatus'].apply(
    lambda x: (x == 'Placed').mean() * 100
).round(1)
for category, rate in cgpa_trends.items():
    print(f"{category}: {rate:.1f}% placement rate")

# ===================================================================
# 8. ADVANCED ANALYTICS INSIGHTS
# ===================================================================

print(f"\n💡 Phase 8: Advanced Analytics Insights")
print("-" * 60)

# Best model selection
results_df = pd.DataFrame(results_advanced)
best_model_idx = results_df['AUC'].idxmax()
best_model_name = results_df.loc[best_model_idx, 'Model']
best_auc = results_df.loc[best_model_idx, 'AUC']

print(f"🏆 Best Performing Model: {best_model_name}")
print(f"🎯 Best AUC Score: {best_auc:.4f}")

# Feature Importance from Random Forest
rf_model = advanced_models['Random Forest']
if 'Random Forest' in [model['Model'] for model in results_advanced]:
    feature_importance = pd.DataFrame({
        'Feature': numeric_features,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(f"\n🌟 Top 5 Most Important Features (Random Forest):")
    for i, (_, row) in enumerate(feature_importance.head().iterrows(), 1):
        print(f"{i}. {row['Feature']}: {row['Importance']:.4f}")

# Predictive Insights
print(f"\n🔮 Predictive Insights for Strategic Decision Making:")
print(f"1. 📚 Academic Excellence: Students with CGPA ≥ 8.0 have {(df_processed[df_processed['CGPA'] >= 8.0]['PlacementStatus'] == 'Placed').mean():.1%} placement rate")
print(f"2. 💼 Experience Matters: Students with ≥2 internships have {(df_processed[df_processed['Internships'] >= 2]['PlacementStatus'] == 'Placed').mean():.1%} placement rate")
print(f"3. 🏆 Skill Development: Students with aptitude ≥85 have {(df_processed[df_processed['AptitudeTestScore'] >= 85]['PlacementStatus'] == 'Placed').mean():.1%} placement rate")

# Success Formula
high_performers = df_processed[
    (df_processed['Performance_Tier'] == 'High_Performer')
]
success_rate = (high_performers['PlacementStatus'] == 'Placed').mean()
print(f"4. 🎯 Success Formula: High Performers achieve {success_rate:.1%} placement rate")

print(f"\n✅ Advanced DMBI Analysis Complete!")
print(f"🚀 Ready for Business Intelligence Dashboard Implementation!")
print("="*80)