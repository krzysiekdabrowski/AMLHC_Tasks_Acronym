import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.inspection import permutation_importance

# randomizing dataset
np.random.seed(42)
X = np.random.rand(1000, 20)
y = np.random.randint(2, size=1000)

# 75%-25% sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(random_state=42))
])

# tuning grid for the RandomForest
param_grid = {
    'rf__n_estimators': [100, 200, 300],
    'rf__max_features': [2, 4, 6, 8]
}

# 10-fold cross-validation
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=10, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_

print("Best parameters found: ", best_params)

best_model = grid_search.best_estimator_

# Evaluate
cv_scores = cross_val_score(best_model, X_train, y_train, cv=10)
print(f"10-fold cross-validation scores: {cv_scores}")
print(f"Mean cross-validation score: {cv_scores.mean()}")

# Feature importance
importances = best_model.named_steps['rf'].feature_importances_
feature_importances = pd.Series(importances, index=[f'Feature {i}' for i in range(X.shape[1])]).sort_values(ascending=False)
print("Feature Importances:")
print(feature_importances)

# Apply the final model to the test set
y_pred = best_model.predict(X_test)

# confusion matrix and performance measures
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)
