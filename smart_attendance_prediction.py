# Smart Attendance Prediction System
# Educational demonstration using a synthetic dataset (not real college records).

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# 1. Load the sample dataset
df = pd.read_csv("smart_attendance_sample_dataset.csv")
print("Dataset shape:", df.shape)
print(df.head())

# 2. Separate features and target
X = df.drop(columns=["low_attendance_risk"])
y = df["low_attendance_risk"]  # 1 = at risk, 0 = not at risk

# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 4. Train the model
model = RandomForestClassifier(
    n_estimators=150, random_state=42, class_weight="balanced"
)
model.fit(X_train, y_train)

# 5. Evaluate
y_pred = model.predict(X_test)
print(f"Test accuracy: {accuracy_score(y_test, y_pred):.3f}")
print("\\nClassification report:")
print(classification_report(y_test, y_pred, target_names=["Not at risk", "At risk"]))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

# 6. Plot feature importance
importance = pd.Series(model.feature_importances_, index=X.columns).sort_values()
importance.plot(kind="barh", title="Feature Importance")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

# 7. Predict for a new example (replace values with valid data)
new_student = pd.DataFrame([{
    "total_classes": 60,
    "attended_classes": 42,
    "current_attendance_pct": 70.0,
    "previous_term_attendance_pct": 78.0,
    "late_arrivals": 5,
    "participation_score_1_to_5": 2
}])
prediction = model.predict(new_student)[0]
probability = model.predict_proba(new_student)[0][1]
print("Prediction:", "AT RISK" if prediction == 1 else "NOT AT RISK")
print(f"Estimated risk probability: {probability:.1%}")
