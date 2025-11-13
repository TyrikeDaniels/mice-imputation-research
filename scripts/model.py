import numpy as np
import pandas as pd

from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

def rf_kfold_eval(path, target, n_splits=5):
    #print("Training!")

    # Read dataframe given path
    df = pd.read_csv(path)

    rf_params = {
        "n_estimators": 130,
        "criterion": "squared_error",
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 5,
        "max_features": 1.0,
        "bootstrap": True,
        "random_state": 42
    }

    # Split features and target
    X = df.drop(columns=[target])
    y = df[target]

    # Initialize Kfold object, lists for record keeping
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    kf_train_scores = []
    kf_val_scores = []

    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = RandomForestRegressor(**rf_params)
        model.fit(X_tr, y_tr)

        yhat_tr = model.predict(X_tr)
        yhat_val = model.predict(X_val)

        kf_train_scores.append(r2_score(y_tr, yhat_tr))
        kf_val_scores.append(r2_score(y_val, yhat_val))

    mean_train_score = np.mean(kf_train_scores)
    mean_val_score = np.mean(kf_val_scores)

    #print(f"Mean Train R^2: {mean_train_score:.4f}, Mean Validation R^2: {mean_val_score:.4f}")

    return (mean_train_score, mean_val_score)
