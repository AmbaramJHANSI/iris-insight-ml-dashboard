from pathlib import Path

import joblib
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.20

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "iris_classifier.pkl"


# Load Dataset
iris = load_iris()

X = iris.data
y = iris.target

feature_names = iris.feature_names
target_names = iris.target_names


# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nLoading Iris Dataset...")
print(f"Training Samples : {len(X_train)}")
print(f"Testing Samples  : {len(X_test)}")


# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=300),
    "Decision Tree": DecisionTreeClassifier(
        random_state=RANDOM_STATE
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        random_state=RANDOM_STATE
    )
}

results = []

best_model = None
best_score = 0


# Train & Evaluate
for model_name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    results.append({
        "Model": model_name,
        "Accuracy": round(accuracy, 4)
    })

    print("\n" + "=" * 60)
    print(model_name.upper())
    print("=" * 60)

    print(f"Accuracy: {accuracy:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    if accuracy > best_score:
        best_score = accuracy
        best_model = model


# Model Comparison
comparison_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(
    comparison_df.sort_values(
        by="Accuracy",
        ascending=False
    )
)

print(f"\nBest Accuracy: {best_score:.4f}")


# Save Best Model
MODEL_DIR.mkdir(exist_ok=True)

joblib.dump(
    best_model,
    MODEL_PATH
)

print(f"\nBest model saved to: {MODEL_PATH}")


# Feature Importance (Random Forest only)
if hasattr(best_model, "feature_importances_"):

    print("\nFeature Importance")
    print("-" * 40)

    for feature, score in zip(
        feature_names,
        best_model.feature_importances_
    ):
        print(
            f"{feature:<25} {score:.4f}"
        )