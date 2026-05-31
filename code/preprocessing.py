

"""
Preprocessing script for lapse and mid-term prediction models.

This script prepares the data for model training by:
- Selecting categorical and numerical predictors
- Splitting the data into train and test sets
- Applying one-hot encoding to categorical variables
- Applying standardization to numerical variables
- Converting the transformed arrays into pandas DataFrames

Note:
The dataset is not included in the repository due to confidentiality constraints.

Required objects before running this script:
- data
- cat_cols
- num_cols
- target_var
"""




# -------------------------------------------------------------------
# Select explanatory variables and target variable
# -------------------------------------------------------------------

X = data[cat_cols + num_cols]
y = data[target_var]


# -------------------------------------------------------------------
# Split the data into training and testing sets
# -------------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -------------------------------------------------------------------
# Define preprocessing pipeline
# -------------------------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", StandardScaler(), num_cols)
    ]
)


# -------------------------------------------------------------------
# Fit encoder/scaler on training data and transform test data
# -------------------------------------------------------------------

X_train_enc = preprocessor.fit_transform(X_train)
X_test_enc = preprocessor.transform(X_test)


# -------------------------------------------------------------------
# Get feature names from the OneHotEncoder
# -------------------------------------------------------------------

cat_feature_names = preprocessor.named_transformers_["cat"].get_feature_names_out(cat_cols)


# -------------------------------------------------------------------
# Combine categorical feature names with numerical column names
# -------------------------------------------------------------------

all_feature_names = list(cat_feature_names) + num_cols


# -------------------------------------------------------------------
# Convert encoded arrays to pandas DataFrames
# -------------------------------------------------------------------

X_train_enc_df = pd.DataFrame(
    X_train_enc,
    columns=all_feature_names,
    index=X_train.index
)

X_test_enc_df = pd.DataFrame(
    X_test_enc,
    columns=all_feature_names,
    index=X_test.index
)
