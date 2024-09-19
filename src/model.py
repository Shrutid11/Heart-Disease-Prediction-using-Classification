import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib


columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']


data = pd.read_csv('data/processed.cleveland.data', header=None, names=columns)

# Replace '?' with NaN
data.replace('?', np.nan, inplace=True)
data = data.apply(pd.to_numeric)


data.fillna(data.mean(), inplace=True)

X = data.drop('target', axis=1)
y = data['target']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Support Vector Machine': SVC()
}


for name, model in models.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"{name} Accuracy: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")

# Train the best model on the entire training set and evaluate on the test set
best_model = GradientBoostingClassifier(random_state=42)
best_model.fit(X_train, y_train)
predictions = best_model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"\nBest Model Test Accuracy: {accuracy * 100:.2f}%")


joblib.dump(best_model, 'model/best_model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')


from sklearn.model_selection import GridSearchCV


param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.05],
    'max_depth': [3, 4, 5]
}


grid_search = GridSearchCV(GradientBoostingClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)


best_model = grid_search.best_estimator_
print(f"Best Parameters: {grid_search.best_params_}")

predictions = best_model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Best Model Test Accuracy: {accuracy * 100:.2f}%")
