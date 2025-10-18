# 🚀 College Placement Data Analysis & Prediction
# Domain: Education Analytics
# Complete implementation of all required steps

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

print("🚀 COLLEGE PLACEMENT PREDICTION ANALYSIS")
print("="*60)

# Step 1: Load the dataset
print("\n📊 Step 1: Loading Dataset...")
df = pd.read_csv('../../data/placementdata.csv')
print(f"✅ Dataset loaded successfully!")
print(f"📐 Shape: {df.shape} (Rows: {df.shape[0]}, Columns: {df.shape[1]})")

# Step 2: Explore the data
print(f"\n🔍 Step 2: Data Exploration...")
print("First 5 rows:")
print(df.head())
print(f"\nDataset Info:")
print(df.info())
print(f"\nMissing values:")
print(df.isnull().sum().sum(), "total missing values")

# Step 3: Data Preprocessing
print(f"\n🔧 Step 3: Data Preprocessing...")
df_processed = df.copy()

# Encode categorical variables
label_encoder = LabelEncoder()
df_processed['PlacementStatus_encoded'] = label_encoder.fit_transform(df_processed['PlacementStatus'])
df_processed['ExtracurricularActivities_encoded'] = label_encoder.fit_transform(df_processed['ExtracurricularActivities'])
df_processed['PlacementTraining_encoded'] = label_encoder.fit_transform(df_processed['PlacementTraining'])

# Select numeric features
numeric_features = ['CGPA', 'Internships', 'Projects', 'Workshops/Certifications', 
                   'AptitudeTestScore', 'SoftSkillsRating', 'SSC_Marks', 'HSC_Marks',
                   'ExtracurricularActivities_encoded', 'PlacementTraining_encoded']

X = df_processed[numeric_features]
y = df_processed['PlacementStatus_encoded']

print(f"✅ Features prepared: {len(numeric_features)} features")
print(f"📊 Target classes: {np.unique(y)} (0=NotPlaced, 1=Placed)")

# Step 4: EDA - Key Statistics
print(f"\n📈 Step 4: Exploratory Data Analysis...")
placed_students = df[df['PlacementStatus'] == 'Placed']
not_placed_students = df[df['PlacementStatus'] == 'NotPlaced']

placement_rate = len(placed_students) / len(df) * 100
print(f"Overall Placement Rate: {placement_rate:.1f}%")

print(f"\nKey Statistics:")
print(f"Average CGPA - Placed: {placed_students['CGPA'].mean():.2f}, Not Placed: {not_placed_students['CGPA'].mean():.2f}")
print(f"Average Internships - Placed: {placed_students['Internships'].mean():.2f}, Not Placed: {not_placed_students['Internships'].mean():.2f}")
print(f"Average Certifications - Placed: {placed_students['Workshops/Certifications'].mean():.2f}, Not Placed: {not_placed_students['Workshops/Certifications'].mean():.2f}")

# Step 5: Split the data
print(f"\n🔄 Step 5: Data Splitting...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training set: {X_train.shape[0]} samples")
print(f"Testing set: {X_test.shape[0]} samples")

# Feature scaling for models that need it
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Train Multiple Models
print(f"\n🤖 Step 6: Training Machine Learning Models...")
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
}

trained_models = {}
predictions = {}

for model_name, model in models.items():
    print(f"Training {model_name}...")
    
    if model_name in ['K-Nearest Neighbors', 'Logistic Regression']:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    trained_models[model_name] = model
    predictions[model_name] = y_pred

print("✅ All models trained successfully!")

# Step 7: Model Evaluation
print(f"\n📊 Step 7: Model Evaluation...")
results = []

for model_name, y_pred in predictions.items():
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    results.append({
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    })
    
    print(f"\n🔍 {model_name}:")
    print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   F1-Score: {f1:.4f}")

# Step 8: Best Model Selection
results_df = pd.DataFrame(results)
best_model_row = results_df.loc[results_df['F1-Score'].idxmax()]
best_model_name = best_model_row['Model']
best_model = trained_models[best_model_name]

print(f"\n🏆 Step 8: Best Model Selection...")
print(f"Best Model: {best_model_name}")
print(f"Best F1-Score: {best_model_row['F1-Score']:.4f}")
print(f"Best Accuracy: {best_model_row['Accuracy']:.4f} ({best_model_row['Accuracy']*100:.2f}%)")

# Step 9: Feature Importance (Random Forest)
print(f"\n🌟 Step 9: Feature Importance Analysis...")
rf_model = trained_models['Random Forest']
feature_importance = rf_model.feature_importances_

importance_df = pd.DataFrame({
    'Feature': numeric_features,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

print("Top 5 Most Important Features:")
for i, (_, row) in enumerate(importance_df.head().iterrows(), 1):
    print(f"{i}. {row['Feature']}: {row['Importance']:.4f}")

# Step 10: Predict for New Students
print(f"\n🎓 Step 10: New Student Predictions...")

def predict_placement(cgpa, internships, projects, certifications, aptitude_score, 
                     soft_skills, ssc_marks, hsc_marks, extracurricular, placement_training):
    extracurricular_encoded = 1 if extracurricular.lower() == 'yes' else 0
    placement_training_encoded = 1 if placement_training.lower() == 'yes' else 0
    
    new_student = np.array([[cgpa, internships, projects, certifications, aptitude_score,
                            soft_skills, ssc_marks, hsc_marks, extracurricular_encoded, 
                            placement_training_encoded]])
    
    if best_model_name in ['K-Nearest Neighbors', 'Logistic Regression']:
        new_student_scaled = scaler.transform(new_student)
        prediction = best_model.predict(new_student_scaled)[0]
        probability = best_model.predict_proba(new_student_scaled)[0]
    else:
        prediction = best_model.predict(new_student)[0]
        probability = best_model.predict_proba(new_student)[0]
    
    return prediction, probability

# Example predictions
print("\nExample Predictions:")

# High performer
prediction1, prob1 = predict_placement(8.5, 2, 3, 3, 90, 4.5, 85, 88, 'Yes', 'Yes')
result1 = "PLACED" if prediction1 == 1 else "NOT PLACED"
confidence1 = prob1[1] if prediction1 == 1 else prob1[0]
print(f"High Performer (CGPA:8.5, 2 internships): {result1} (Confidence: {confidence1:.2%})")

# Average performer
prediction2, prob2 = predict_placement(7.2, 1, 2, 1, 75, 4.0, 70, 72, 'No', 'Yes')
result2 = "PLACED" if prediction2 == 1 else "NOT PLACED"
confidence2 = prob2[1] if prediction2 == 1 else prob2[0]
print(f"Average Performer (CGPA:7.2, 1 internship): {result2} (Confidence: {confidence2:.2%})")

# Below average performer
prediction3, prob3 = predict_placement(6.8, 0, 1, 0, 65, 3.5, 60, 65, 'No', 'No')
result3 = "PLACED" if prediction3 == 1 else "NOT PLACED"
confidence3 = prob3[1] if prediction3 == 1 else prob3[0]
print(f"Below Average (CGPA:6.8, 0 internships): {result3} (Confidence: {confidence3:.2%})")

# Business Insights
print(f"\n💡 Business Analytics Insights...")
print("="*50)

# CGPA Analysis
placed_cgpa = df[df['PlacementStatus'] == 'Placed']['CGPA'].mean()
not_placed_cgpa = df[df['PlacementStatus'] == 'NotPlaced']['CGPA'].mean()
print(f"1. CGPA Impact: Placed students have {placed_cgpa - not_placed_cgpa:.2f} higher CGPA on average")

# Internship Analysis
with_internships = df[df['Internships'] > 0]['PlacementStatus']
without_internships = df[df['Internships'] == 0]['PlacementStatus']
with_rate = (with_internships == 'Placed').mean() * 100
without_rate = (without_internships == 'Placed').mean() * 100
multiplier = with_rate / without_rate if without_rate > 0 else 0
print(f"2. Internship Impact: Students with internships have {multiplier:.1f}x higher placement rate")

# Certification Analysis
high_cert = df[df['Workshops/Certifications'] >= 2]['PlacementStatus']
low_cert = df[df['Workshops/Certifications'] < 2]['PlacementStatus']
high_cert_rate = (high_cert == 'Placed').mean() * 100
low_cert_rate = (low_cert == 'Placed').mean() * 100
print(f"3. Certification Impact: ≥2 certifications: {high_cert_rate:.1f}% vs <2: {low_cert_rate:.1f}%")

# Success Factors
success_criteria = (
    (df['CGPA'] >= 8.0) & 
    (df['Internships'] >= 1) & 
    (df['Workshops/Certifications'] >= 2)
)
success_rate = (df[success_criteria]['PlacementStatus'] == 'Placed').mean() * 100
print(f"4. Combined Success: CGPA≥8.0 + ≥1 internship + ≥2 certs = {success_rate:.1f}% placement rate")

# Final Summary
print(f"\n🎯 PROJECT SUMMARY")
print("="*50)
print(f"📊 Total Students Analyzed: {len(df):,}")
print(f"🏆 Best Model: {best_model_name}")
print(f"🎯 Model Accuracy: {best_model_row['Accuracy']:.1%}")
print(f"📈 Overall Placement Rate: {placement_rate:.1f}%")
print(f"🌟 Top Success Factor: {importance_df.iloc[0]['Feature']}")
print(f"✅ Project Status: COMPLETED SUCCESSFULLY!")

print(f"\n🚀 Ready for real-world deployment!")
print("="*60)