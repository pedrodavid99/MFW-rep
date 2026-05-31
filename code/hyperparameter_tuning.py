"""
Hyperparameter tuning script for lapse prediction models.

This file contains the Optuna tuning procedures used for the lapse models.
The same tuning structure was applied to the mid-term models.

Models included:
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- LightGBM
- Multi-Layer Perceptron

Required objects before running this script:
- X_train_enc_df
- y_train

Note:
The dataset is not included in the repository due to confidentiality constraints.
"""

import optuna
import xgboost as xgb

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier


# -------------------------------------------------------------------
# General tuning settings
# -------------------------------------------------------------------

optuna.logging.set_verbosity(optuna.logging.WARNING)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# -------------------------------------------------------------------
# Logistic Regression tuning
# -------------------------------------------------------------------

def objective_lr(trial):
    penalty = trial.suggest_categorical(
        "penalty",
        ["l1", "l2", "elasticnet"]
    )

    C = trial.suggest_float(
        "C",
        0.01,
        10.0,
        log=True
    )

    weight = trial.suggest_categorical(
        "class_weight_pos",
        [None, 3, 5, 8, 12]
    )

    class_weight = None if weight is None else {0: 1, 1: weight}

    l1_ratio = (
        trial.suggest_float("l1_ratio", 0.0, 1.0)
        if penalty == "elasticnet"
        else None
    )

    model = LogisticRegression(
        C=C,
        penalty=penalty,
        solver="saga",
        l1_ratio=l1_ratio,
        class_weight=class_weight,
        max_iter=500,
        tol=1e-3,
        random_state=42,
        warm_start=False
    )

    scores = cross_val_score(
        model,
        X_train_enc_df,
        y_train,
        cv=cv,
        scoring="neg_brier_score",
        n_jobs=-1
    )

    return scores.mean()


study_lr = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42)
)

study_lr.optimize(
    objective_lr,
    n_trials=20,
    show_progress_bar=True
)

p = study_lr.best_params

print(f"Best Brier Score: {-study_lr.best_value:.4f}")
print(f"Best hyperparameters: {p}")

best_lr = LogisticRegression(
    C=p["C"],
    penalty=p["penalty"],
    solver="saga",
    l1_ratio=p.get("l1_ratio", None),
    class_weight=None if p["class_weight_pos"] is None else {0: 1, 1: p["class_weight_pos"]},
    max_iter=500,
    tol=1e-3,
    random_state=42
)

best_lr.fit(X_train_enc_df, y_train)


# -------------------------------------------------------------------
# Decision Tree tuning
# -------------------------------------------------------------------

def objective_dt(trial):
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_samples_split": trial.suggest_int("min_samples_split", 10, 200),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 100),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
        "random_state": 42
    }

    model = DecisionTreeClassifier(**params)

    scores = cross_val_score(
        model,
        X_train_enc_df,
        y_train,
        cv=cv,
        scoring="neg_brier_score",
        n_jobs=-1
    )

    return scores.mean()


study_dt = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42)
)

study_dt.optimize(
    objective_dt,
    n_trials=20,
    show_progress_bar=True
)

print(f"\nBest Brier Score: {-study_dt.best_value:.4f}")
print(f"Best hyperparameters:\n{study_dt.best_params}")

p = study_dt.best_params

best_dt = DecisionTreeClassifier(**p)
best_dt.fit(X_train_enc_df, y_train)


# -------------------------------------------------------------------
# Random Forest tuning
# -------------------------------------------------------------------

def objective_rf(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "min_samples_split": trial.suggest_int("min_samples_split", 10, 200),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 100),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        "max_samples": trial.suggest_float("max_samples", 0.6, 0.9),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
        "random_state": 42,
        "n_jobs": -1
    }

    model = RandomForestClassifier(**params)

    scores = cross_val_score(
        model,
        X_train_enc_df,
        y_train,
        cv=cv,
        scoring="neg_brier_score",
        n_jobs=-1
    )

    return scores.mean()


study_rf = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42)
)

study_rf.optimize(
    objective_rf,
    n_trials=20,
    show_progress_bar=True
)

print(f"\nBest Brier Score: {-study_rf.best_value:.4f}")
print(f"Best hyperparameters:\n{study_rf.best_params}")

best_rf = RandomForestClassifier(**study_rf.best_params)
best_rf.fit(X_train_enc_df, y_train)


# -------------------------------------------------------------------
# XGBoost tuning
# -------------------------------------------------------------------

scale_pos = (y_train == 0).sum() / (y_train == 1).sum()

def objective_xgb(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 5.0, log=True),
        "scale_pos_weight": scale_pos,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0
    }

    model = xgb.XGBClassifier(**params)

    scores = cross_val_score(
        model,
        X_train_enc_df,
        y_train,
        cv=cv,
        scoring="neg_brier_score",
        n_jobs=-1
    )

    return scores.mean()


study_xgb = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42)
)

study_xgb.optimize(
    objective_xgb,
    n_trials=20,
    show_progress_bar=True
)

print(f"\nBest Brier Score: {-study_xgb.best_value:.4f}")
print(f"Best hyperparameters:\n{study_xgb.best_params}")

best_xgb = xgb.XGBClassifier(
    **study_xgb.best_params,
    scale_pos_weight=scale_pos,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
    verbosity=0
)

best_xgb.fit(X_train_enc_df, y_train)


# -------------------------------------------------------------------
# LightGBM tuning
# -------------------------------------------------------------------

scale_pos = (y_train == 0).sum() / (y_train == 1).sum()

def objective_lgbm(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 5.0, log=True),
        "scale_pos_weight": scale_pos,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1
    }

    model = LGBMClassifier(**params)

    scores = cross_val_score(
        model,
        X_train_enc_df,
        y_train,
        cv=cv,
        scoring="neg_brier_score",
        n_jobs=-1
    )

    return scores.mean()


study_lgbm = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42)
)

study_lgbm.optimize(
    objective_lgbm,
    n_trials=20,
    show_progress_bar=True
)

print(f"\nBest Brier Score: {-study_lgbm.best_value:.4f}")
print(f"Best hyperparameters:\n{study_lgbm.best_params}")

best_lgbm = LGBMClassifier(
    **study_lgbm.best_params,
    scale_pos_weight=scale_pos,
    random_state=42,
    n_jobs=-1,
    verbosity=-1
)

best_lgbm.fit(X_train_enc_df, y_train)


# -------------------------------------------------------------------
# Multi-Layer Perceptron tuning
# -------------------------------------------------------------------

def objective_mlp(trial):
    n_layers = trial.suggest_int("n_layers", 1, 3)
    layer_size = trial.suggest_int("layer_size", 32, 256, step=32)

    hidden_layer_sizes = tuple([layer_size] * n_layers)

    params = {
        "hidden_layer_sizes": hidden_layer_sizes,
        "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
        "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
        "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256])
    }

    model = MLPClassifier(
        **params,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42
    )

    cv_mlp = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42
    )

    scores = cross_val_score(
        model,
        X_train_enc_df,
        y_train,
        cv=cv_mlp,
        scoring="neg_brier_score",
        n_jobs=-1
    )

    return scores.mean()


study_mlp = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42)
)

study_mlp.optimize(
    objective_mlp,
    n_trials=30,
    show_progress_bar=True
)

print(f"\nBest Brier Score: {-study_mlp.best_value:.4f}")
print(f"Best hyperparameters:\n{study_mlp.best_params}")

best_mlp_params = study_mlp.best_params.copy()

n_layers = best_mlp_params.pop("n_layers")
layer_size = best_mlp_params.pop("layer_size")

best_mlp_params["hidden_layer_sizes"] = tuple([layer_size] * n_layers)

best_mlp = MLPClassifier(
    **best_mlp_params,
    max_iter=500,
    random_state=42
)

best_mlp.fit(X_train_enc_df, y_train)
