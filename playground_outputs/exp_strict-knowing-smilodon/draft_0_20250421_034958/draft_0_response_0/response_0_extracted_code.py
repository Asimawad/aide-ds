import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split

# Load data
train = pd.read_csv("input/train.csv")
test = pd.read_csv("input/test.csv")
sample_submission = pd.read_csv("input/sample_submission.csv")

# Prepare data
X = train[["EAP", "HPL", "MWS"]]
y = train["author"]

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize model
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

# Train model
rf.fit(X_train, y_train)

# Evaluate model
y_pred = rf.predict(X_val)
loss = log_loss(y_val, y_pred)
print(f"Validation Loss: {loss}")

# Generate predictions for test set
y_pred_test = rf.predict(X_test)
y_pred_test_proba = rf.predict_proba(X_test)

# Save predictions
submission_path = "submission/submission.csv"
pd.DataFrame({"id": X_test["id"], "author": y_pred_test}).to_csv(
    submission_path, index=False
)
