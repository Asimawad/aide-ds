import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RandomForestClassifier
from sklearn.metrics import multiclassification_loss
from sklearn.model_selection import train_test_split

# Load data
train = pd.read_csv("input/train.csv")
test = pd.read_csv("input/test.csv")
sample_submission = pd.read_csv("input/sample_submission.csv")

# Prepare features
text_features = ["text"]
vectorizer = TfidfVectorizer(max_features=1000)
text_features = vectorizer.fit_transform(text_features)

# Combine features
train = pd.concat([train, pd.DataFrame(text_features)], axis=1)
test = pd.concat([test, pd.DataFrame(text_features)], axis=1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    train, train["author"], test_size=0.2, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# Model
model = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42)

# Train
model.fit(X_train, y_train)

# Evaluate
val_pred = model.predict(X_val)
print(f"Validation Loss: {multiclassification_loss(model, y_val, labels=3)}")

# Save predictions
submission = pd.DataFrame({"id": test["id"], "author": model.predict(X_test)})
submission.to_csv("submission/submission.csv", index=False)
