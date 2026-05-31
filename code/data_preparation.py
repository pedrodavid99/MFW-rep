"""
Generic data preparation and missing-value imputation utilities.

This file contains non-confidential examples of data preparation steps used in
the analysis, including:
- postcode-prefix mode imputation
- age-bin based imputation for biometric-style variables
- derived ratio calculation

Confidential note:
Original dataset names, business variable names, and internal feature labels
have been masked/anonymized. The logic is preserved, but variable names have
been replaced with generic placeholders to avoid disclosing sensitive
information.
"""

import numpy as np
import pandas as pd


# -------------------------------------------------------------------
# Postcode-prefix mode imputation
# -------------------------------------------------------------------

def fillna_by_postcode_mode_vectorized(
    df,
    cols,
    postcode_col="MASKED_POSTCODE_COLUMN"
):
    """
    Fill missing values using the mode of each variable within progressively
    shorter postcode prefixes.

    The function first tries to impute using longer postcode prefixes and then
    progressively shorter prefixes if no value is available.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.

    cols : list
        List of columns to impute.

    postcode_col : str
        Column containing postcode information. The original column name has
        been masked for confidentiality.

    Returns
    -------
    pandas.DataFrame
        Dataframe with imputed values.
    """

    df = df.copy()

    for col in cols:
        filled = pd.Series([False] * len(df), index=df.index)

        for n in [9, 8, 7, 5, 4, 3, 2, 1]:
            prefix = df[postcode_col].astype(str).str[:n]

            mode_map = (
                df.loc[~df[col].isna()]
                .groupby(prefix)[col]
                .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
            )

            mask = df[col].isna() & ~filled
            fill_prefix = prefix[mask]
            fill_values = fill_prefix.map(mode_map)

            df.loc[mask, col] = fill_values
            filled = filled | (mask & fill_values.notna())

    return df


# -------------------------------------------------------------------
# Age-bin based imputation for biometric-style variables
# -------------------------------------------------------------------

def impute_masked_biometric_variables(df):
    """
    Impute masked biometric-style variables using age bins.

    This function is a confidentiality-preserving version of the original
    preprocessing logic. Original variable names have been anonymized.

    Masked variable mapping:
    - MASKED_WEIGHT_VAR: original weight-like variable
    - MASKED_HEIGHT_VAR: original height-like variable
    - MASKED_AGE_VAR: original age variable
    - MASKED_DERIVED_RATIO_VAR: derived ratio variable, equivalent to
      weight divided by height squared

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.

    Returns
    -------
    pandas.DataFrame
        Dataframe with imputed biometric-style variables and derived ratio.
    """

    df = df.copy()

    # Convert masked biometric variables to float
    df["MASKED_WEIGHT_VAR"] = df["MASKED_WEIGHT_VAR"].astype(float)
    df["MASKED_HEIGHT_VAR"] = df["MASKED_HEIGHT_VAR"].astype(float)

    # Drop old masked height column if it exists
    if "MASKED_OLD_HEIGHT_VAR" in df.columns:
        df = df.drop(columns=["MASKED_OLD_HEIGHT_VAR"])

    # Define age bins and labels
    bins = [0, 1, 3, 5, 6, 10, 15, 20, 30, 60, 100]
    labels = [
        "0-1",
        "2-3",
        "4-5",
        "5-6",
        "7-10",
        "11-15",
        "16-20",
        "21-30",
        "31-60",
        "61-100"
    ]

    # Assign age bins
    df["MASKED_AGE_BIN"] = pd.cut(
        df["MASKED_AGE_VAR"],
        bins=bins,
        labels=labels,
        right=True,
        include_lowest=True
    )

    # Compute valid bin-level means
    valid = df[
        (df["MASKED_WEIGHT_VAR"].notna()) &
        (df["MASKED_WEIGHT_VAR"] > 0) &
        (df["MASKED_WEIGHT_VAR"] <= 150) &
        (df["MASKED_HEIGHT_VAR"].notna()) &
        (df["MASKED_HEIGHT_VAR"] > 0) &
        (df["MASKED_HEIGHT_VAR"] <= 2.2)
    ]

    bin_means = valid.groupby("MASKED_AGE_BIN").agg({
        "MASKED_WEIGHT_VAR": "mean",
        "MASKED_HEIGHT_VAR": "mean"
    })

    # Impute masked weight variable
    for bin_label in labels:
        mask = (
            (df["MASKED_AGE_BIN"] == bin_label) &
            (
                df["MASKED_WEIGHT_VAR"].isna() |
                (df["MASKED_WEIGHT_VAR"] == 0) |
                (df["MASKED_WEIGHT_VAR"] > 150)
            )
        )

        mean_weight = (
            bin_means.loc[bin_label, "MASKED_WEIGHT_VAR"]
            if bin_label in bin_means.index
            else np.nan
        )

        df.loc[mask, "MASKED_WEIGHT_VAR"] = mean_weight

    # Impute masked height variable
    for bin_label in labels:
        mask = (
            (df["MASKED_AGE_BIN"] == bin_label) &
            (
                df["MASKED_HEIGHT_VAR"].isna() |
                (df["MASKED_HEIGHT_VAR"] == 0) |
                (df["MASKED_HEIGHT_VAR"] > 2.2)
            )
        )

        mean_height = (
            bin_means.loc[bin_label, "MASKED_HEIGHT_VAR"]
            if bin_label in bin_means.index
            else np.nan
        )

        df.loc[mask, "MASKED_HEIGHT_VAR"] = mean_height

    # Calculate masked derived ratio
    df["MASKED_DERIVED_RATIO_VAR"] = (
        df["MASKED_WEIGHT_VAR"] /
        (df["MASKED_HEIGHT_VAR"] ** 2)
    )

    # Fill remaining missing derived ratio values with the mean
    df["MASKED_DERIVED_RATIO_VAR"] = df["MASKED_DERIVED_RATIO_VAR"].fillna(
        df["MASKED_DERIVED_RATIO_VAR"].mean()
    )

    return df
