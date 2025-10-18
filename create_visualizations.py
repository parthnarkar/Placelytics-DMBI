import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Load and preprocess data
print("📊 Creating visualizations for placement analysis...")
df = pd.read_csv('placementdata.csv')

# Preprocessing
df_processed = df.copy()
label_encoder = LabelEncoder()
df_processed['PlacementStatus_encoded'] = label_encoder.fit_transform(df_processed['PlacementStatus'])
df_processed['ExtracurricularActivities_encoded'] = label_encoder.fit_transform(df_processed['ExtracurricularActivities'])
df_processed['PlacementTraining_encoded'] = label_encoder.fit_transform(df_processed['PlacementTraining'])

numeric_features = ['CGPA', 'Internships', 'Projects', 'Workshops/Certifications', 
                   'AptitudeTestScore', 'SoftSkillsRating', 'SSC_Marks', 'HSC_Marks',
                   'ExtracurricularActivities_encoded', 'PlacementTraining_encoded']

X = df_processed[numeric_features]
y = df_processed['PlacementStatus_encoded']

# Train Random Forest for feature importance
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

# Create comprehensive visualization
fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# 1. Placement Status Distribution
placement_counts = df['PlacementStatus'].value_counts()
colors = ['#FF6B6B', '#4ECDC4']
axes[0, 0].pie(placement_counts.values, labels=placement_counts.index, autopct='%1.1f%%', 
               colors=colors, startangle=90)
axes[0, 0].set_title('📊 Placement Status Distribution', fontsize=14, fontweight='bold')

# 2. CGPA vs Placement
sns.boxplot(data=df, x='PlacementStatus', y='CGPA', palette=['#FF6B6B', '#4ECDC4'], ax=axes[0, 1])
axes[0, 1].set_title('📚 CGPA vs Placement Status', fontsize=14, fontweight='bold')

# 3. Internships vs Placement Rate
internship_placement = pd.crosstab(df['Internships'], df['PlacementStatus'], normalize='index') * 100
internship_placement.plot(kind='bar', color=['#FF6B6B', '#4ECDC4'], ax=axes[0, 2])
axes[0, 2].set_title('💼 Internships vs Placement Rate', fontsize=14, fontweight='bold')
axes[0, 2].set_xlabel('Number of Internships')
axes[0, 2].set_ylabel('Percentage (%)')
axes[0, 2].legend(title='Placement Status')
axes[0, 2].tick_params(axis='x', rotation=0)

# 4. Feature Importance
feature_importance = rf_model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': numeric_features,
    'Importance': feature_importance
}).sort_values('Importance', ascending=True)

bars = axes[1, 0].barh(importance_df['Feature'], importance_df['Importance'], color='#96CEB4', alpha=0.8)
axes[1, 0].set_title('🌟 Feature Importance (Random Forest)', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Importance Score')

# Add value labels
for bar, imp in zip(bars, importance_df['Importance']):
    axes[1, 0].text(imp + 0.005, bar.get_y() + bar.get_height()/2, 
                    f'{imp:.3f}', ha='left', va='center', fontweight='bold')

# 5. Correlation Heatmap
correlation_matrix = df_processed[numeric_features + ['PlacementStatus_encoded']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='RdYlBu_r', center=0, 
            square=True, linewidths=0.5, ax=axes[1, 1])
axes[1, 1].set_title('🔥 Feature Correlation Heatmap', fontsize=14, fontweight='bold')

# 6. Success Factors Comparison
factors = ['Overall', 'CGPA≥8.0', '≥1 Internship', '≥2 Certs', 'Combined Success']
rates = [
    (df['PlacementStatus'] == 'Placed').mean() * 100,
    (df[df['CGPA'] >= 8.0]['PlacementStatus'] == 'Placed').mean() * 100,
    (df[df['Internships'] >= 1]['PlacementStatus'] == 'Placed').mean() * 100,
    (df[df['Workshops/Certifications'] >= 2]['PlacementStatus'] == 'Placed').mean() * 100,
    (df[(df['CGPA'] >= 8.0) & (df['Internships'] >= 1) & (df['Workshops/Certifications'] >= 2)]['PlacementStatus'] == 'Placed').mean() * 100
]

colors_factors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
bars_factors = axes[1, 2].bar(factors, rates, color=colors_factors, alpha=0.8)
axes[1, 2].set_title('📈 Placement Rates by Success Factors', fontsize=14, fontweight='bold')
axes[1, 2].set_ylabel('Placement Rate (%)')
axes[1, 2].set_ylim(0, 100)
axes[1, 2].tick_params(axis='x', rotation=45)

# Add value labels
for bar, rate in zip(bars_factors, rates):
    axes[1, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('placement_analysis_visualizations.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Visualizations saved as 'placement_analysis_visualizations.png'")
print("🎨 Complete visual analysis ready for presentation!")