

"""
Libraries used in the analysis.

This file lists the Python libraries used for:
- Data manipulation
- Data preprocessing
- Machine learning models
- Calibration
- Model evaluation
- Visualization
- SHAP explainability
- Hyperparameter optimization
"""

# Data manipulation
import numpy as np
import pandas as pd
from collections import Counter

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Preprocessing and pipelines
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Train-test split and cross-validation
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
    cross_val_score
)

# Machine learning models
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

import xgboost as xgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Model evaluation metrics
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    average_precision_score,
    classification_report,
    RocCurveDisplay,
    ConfusionMatrixDisplay,
    brier_score_loss
)

# Calibration
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import (
    CalibratedClassifierCV,
    CalibrationDisplay
)

# Model utilities
from sklearn.base import clone

# Explainability
import shap

# Hyperparameter optimization
import optuna
from optuna import visualization as vis
