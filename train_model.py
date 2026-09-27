# train_model.py
# This script cleans the diabetes dataset, trains a Logistic Regression
# model, checks how good it is, and saves it so the website can use it.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pickle

# ---------- Step 1: Load the dataset ----------
data = pd.read_csv("data/diabetes.csv")
print("Dataset loaded. Shape:", data.shape)

# ---------- Step 2: Clean the data ----------
# In this dataset, a value of 0 in these columns doesn't make medical sense
# (a real person can't have 0 glucose or 0 BMI). So a 0 here actually means
# "this value is missing", not that it's really zero.
columns_with_invalid_zeros = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

for col in columns_with_invalid_zeros:
    # Replace 0 with the median of that column (ignoring the 0s)
    median_value = data[data[col] != 0][col].median()
    data[col] = data[col].replace(0, median_value)

print("Missing/invalid zero values replaced with column medians.")

# ---------- Step 3: Split features (X) and target (y) ----------
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# ---------- Step 4: Scale the features ----------
# Logistic Regression works better when all features are on a similar scale.
# For example, Age (21-81) and BMI (18-67) are on very different ranges.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------- Step 5: Split into training and testing sets ----------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ---------- Step 6: Train the Logistic Regression model ----------
model = LogisticRegression()
model.fit(X_train, y_train)

# ---------- Step 7: Evaluate the model ----------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("\n--- Model Performance ---")
print(f"Accuracy:  {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall:    {recall:.2f}")

# ---------- Step 8: Save the model and the scaler ----------
# We need to save the scaler too, because new user input must be scaled
# the same way the training data was scaled.
with open("model/diabetes_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\nModel and scaler saved to the 'model' folder.")