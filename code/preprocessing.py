# libraries used
import numpy as np 
import pandas as pd 
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix, average_precision_score
from sklearn.metrics import classification_report, RocCurveDisplay, ConfusionMatrixDisplay
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from lightgbm import LGBMClassifier
import shap
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.isotonic import IsotonicRegression
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from sklearn.calibration import CalibrationDisplay
import optuna
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_score
from optuna import visualization as vis
from sklearn.neural_network import MLPClassifier

################################################################################################### spliting the data into train and test and aply the encoder + standartization###################################################################################################
## after importing the dataset and data preprocessing
X= data[cat_cols + num_cols]
y =data[target_var]


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", StandardScaler(), num_cols)
    ]
)



X_train_enc = preprocessor.fit_transform(X_train)
X_test_enc  = preprocessor.transform(X_test)




# Get feature names for categorical columns from the OneHotEncoder
cat_feature_names = preprocessor.named_transformers_["cat"].get_feature_names_out(cat_cols)

# Combine with numerical column names
all_feature_names = list(cat_feature_names) + num_cols

# Convert the encoded arrays to DataFrames
X_train_enc_df = pd.DataFrame(X_train_enc, columns=all_feature_names, index=X_train.index)
X_test_enc_df = pd.DataFrame(X_test_enc, columns=all_feature_names, index=X_test.index)











