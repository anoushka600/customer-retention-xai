import pandas as pd
import xgboost as xgb
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("data.csv")

# Split features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# One-hot encoding
X = pd.get_dummies(X)

# ✅ TRAIN/TEST SPLIT (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05
)

# ✅ Fit on the training data only
model.fit(X_train, y_train)

# ✅ Test the model and print the score
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model trained! Testing Accuracy: {accuracy * 100:.2f}%")

# Save model and columns for the dashboard
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(X.columns.tolist(), open("columns.pkl", "wb"))

print("Model and columns saved successfully!")