import pandas as pd
import numpy as np
import shap
import lime
import lime.lime_tabular
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
import joblib 

# Loading Data from csv file
df = pd.read_csv('heart_cleaned.csv')

# Data Exploration & Overview
print("Dataset Overview:")
print(df.info())
print("\nMissing values:")
print(df.isnull().sum())

# Visualizing target variable distribution;useful for class balancing
plt.figure(figsize=(8, 5))
sns.countplot(x='target', data=df)
plt.title('Target Variable Distribution')
plt.show()

# Correlation heatmap showing correlations between different features of the dataset.
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Feature Correlation Heatmap")
plt.show()

# Preprocessing the Data
X = df.drop('target', axis=1)
y = df['target']

# Normalizing the Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Using SMOTE for dataset balancing.
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Dictionary to store the model evaluation metrics
results = {}

#  Training Multiple Models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, solver='liblinear'),
    'Random Forest': RandomForestClassifier(random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
    'SVM': SVC(probability=True, random_state=42),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    
    # Calculating metrics for training set
    train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_pred)
    
    # Calculating metrics for test set
    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n{name} Performance:")
    print(f"Train Accuracy: {train_accuracy:.2f}")
    print(f"Test Accuracy: {test_accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    results[name] = {
        'model': model,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


# Checking and Comparing all model performance
results_df = pd.DataFrame(results).T
print("\nModel Comparison:\n", results_df)

#The Feature Importance (Permutation Importance for Best Model)
best_model = results['Random Forest']['model']  # we are Assuming Random Forest was selected for importance
perm_importance = permutation_importance(best_model, X_test, y_test, scoring='accuracy')
feature_importance = perm_importance.importances_mean
feature_names = df.drop(columns=['target']).columns
sorted_idx = np.argsort(feature_importance)[::-1]

plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_importance)), feature_importance[sorted_idx], align='center')
plt.xticks(range(len(feature_importance)), feature_names[sorted_idx], rotation=90)
plt.title('Permutation Feature Importance')
plt.tight_layout()
plt.show()

#  XAI,SHAP & LIME
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, show=False)
plt.title('SHAP Summary Plot')
plt.show()

lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train,
    mode="classification",
    feature_names=feature_names,
    class_names=["No Heart Disease", "Heart Disease"],
    discretize_continuous=True
)

# Explaining an instance
exp = lime_explainer.explain_instance(X_test[0], best_model.predict_proba)
fig = exp.as_pyplot_figure()
plt.title('LIME Explanation for Instance')
plt.show()

#Analyzing the Misclassified Cases
misclassified_indices = np.where(best_model.predict(X_test) != y_test)[0]
misclassified_data = X_test[misclassified_indices]
print("\nMisclassified Data Analysis:")
print(misclassified_data)

#Saving the Best Model
joblib.dump(best_model, 'best_heart_disease_model.joblib')
print("\nModel saved as 'best_heart_disease_model.joblib'")