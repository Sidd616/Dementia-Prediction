import numpy as np
import pandas as pd
import pytest

from preprocessing import LEAKAGE_COLUMNS, DementiaPreprocessor


def make_frame():
    return pd.DataFrame({
        'Diabetic': [0, 1, 0, 1],
        'AlcoholLevel': [0.1, 0.2, np.nan, 0.4],
        'HeartRate': [60, 70, 80, np.nan],
        'Age': [50, 60, 70, 80],
        'Prescription': ['Donepezil', '', '', 'Memantine'],
        'Dosage in mg': [10, '', '', 20],
        'Medication_History': ['Yes', 'No', 'No', 'Yes'],
        'Dominant_Hand': ['Right', 'Left', 'Right', 'Left'],
        'Gender': ['Male', 'Female', 'Male', 'Female'],
        'Family_History': ['Yes', 'No', 'No', 'Yes'],
        'Smoking_Status': ['Never Smoked', 'Former Smoker', 'Never Smoked', 'Current Smoker'],
        'APOE_ε4': ['Positive', 'Negative', 'Negative', 'Positive'],
        'Physical_Activity': ['Sedentary', 'Mild Activity', 'Moderate Activity', 'High Activity'],
        'Depression_Status': ['No', 'Yes', 'No', 'No'],
        'Nutrition_Diet': ['Balanced Diet', 'Low-Carb Diet', 'Other', 'Mediterranean Diet'],
        'Sleep_Quality': ['Good', 'Fair', 'Poor', 'Good'],
        'Chronic_Health_Conditions': ['None', 'Heart Disease', 'Diabetes', 'Hypertension'],
    })


def test_leakage_columns_are_dropped():
    pre = DementiaPreprocessor()
    X = pre.fit_transform(make_frame())
    assert not any(c in pre.feature_columns for c in LEAKAGE_COLUMNS)
    assert X.shape[1] == len(pre.feature_columns)


def test_numeric_missing_values_use_median_not_zero():
    frame = make_frame()
    pre = DementiaPreprocessor()
    X = pre.fit_transform(frame)

    heart_rate_idx = pre.feature_columns.index('HeartRate')
    median_imputed = frame['HeartRate'].fillna(frame['HeartRate'].median()).astype(float).values
    expected_scaled = (median_imputed[3] - median_imputed.mean()) / median_imputed.std(ddof=0)
    assert X[3, heart_rate_idx] == pytest.approx(expected_scaled, rel=1e-6)

    # Make sure we're not back to the old constant-0 imputation.
    zero_imputed = frame['HeartRate'].fillna(0).astype(float).values
    expected_if_zero_imputed = (0 - zero_imputed.mean()) / zero_imputed.std(ddof=0)
    assert X[3, heart_rate_idx] != pytest.approx(expected_if_zero_imputed, rel=1e-6)


def test_transform_handles_unseen_category_without_crashing():
    frame = make_frame()
    pre = DementiaPreprocessor()
    pre.fit_transform(frame)

    new_row = frame.iloc[[0]].copy()
    new_row['Gender'] = 'NonBinary'  # unseen at fit time
    X = pre.transform(new_row)
    assert X.shape == (1, len(pre.feature_columns))


def test_transform_before_fit_raises():
    pre = DementiaPreprocessor()
    with pytest.raises(RuntimeError):
        pre.transform(make_frame())
