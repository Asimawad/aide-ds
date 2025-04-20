import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import os

# Load data
train_data = pd.read_csv(os.path.join("input", "train.csv"))
test_data = pd.read_csv(os.path.join("input", "test.csv"))
submission_data = pd.read_csv(os.path.join("submission", "submission.csv"))

# Load target variable
y = train_data["SalePrice"]
X = train_data.drop("SalePrice", axis=1)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Feature selection (if needed)
# selected_features = ['GrLivArea', 'Kima', 'YrSold', 'Full bath', 'SaleType', 'YrConcat', 'SalePrice']
# X_train_selected = selected_features
# X_val_selected = selected_features

# Model selection and hyperparameter tuning
ridge = Ridge(alpha=1.0, max_iter=100)
param_grid = {
    "alpha": [0.1, 1.0, 10.0],
    "fit_intercept": [False, True],
    "tolerance": [0.0001, 0.001],
}
grid_search = GridSearchCV(ridge, param_grid, scoring="neg_mean_squared_error", cv=5)
grid_search.fit(X_train_scaled, y_train)

best_ridge = grid_search.best_estimator_
best_params = grid_search.best_params_

# Model training
y_pred = best_ridge.predict(X_val_scaled)

# Model evaluation
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"RMSE: {rmse}")

# Make predictions on test data
y_test = best_ridge.predict(X_test_scaled)
y_pred_test = best_ridge.predict(X_test)

# Prepare submission file
submission_data = pd.concat(
    [submission_data, pd.DataFrame({"Id": test_data["Id"], "SalePrice": y_pred_test})],
    axis=1,
)
submission_data.to_csv(os.path.join("submission", "submission.csv"), index=False)

# Final evaluation
print(f"Final RMSE: {rmse}")
