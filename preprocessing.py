import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

CATEGORICAL_COLUMNS = [
    'Dominant_Hand', 'Gender', 'Family_History',
    'Smoking_Status', 'APOE_ε4', 'Physical_Activity', 'Depression_Status',
    'Nutrition_Diet', 'Sleep_Quality', 'Chronic_Health_Conditions'
]

# Prescription / Dosage in mg / Medication_History are only ever populated for
# patients already on anti-dementia medication, i.e. already diagnosed. They
# perfectly encode the target (verified: every non-null Prescription row has
# Dementia == 1 and vice versa) and must be excluded as predictive features.
LEAKAGE_COLUMNS = ['Prescription', 'Dosage in mg', 'Medication_History']


class DementiaPreprocessor:
    """Fits categorical encoders, a numeric imputer, and a scaler on training
    data, then applies the same fitted transforms to unseen data. Call
    fit_transform() exactly once (on the training split only), then transform()
    for validation/test/inference data.
    """

    def __init__(self):
        self.label_encoders = {}
        self.category_mappings = {}
        self.numeric_imputer = SimpleImputer(strategy='median')
        self.scaler = StandardScaler()
        self.numerical_columns = None
        self.feature_columns = None
        self.fitted = False

    def fit_transform(self, data):
        return self._process(data, fit=True)

    def transform(self, data):
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fit before calling transform().")
        return self._process(data, fit=False)

    def _process(self, data, fit):
        data = data.drop(columns=[c for c in LEAKAGE_COLUMNS if c in data.columns]).copy()

        if self.numerical_columns is None:
            self.numerical_columns = [c for c in data.columns if c not in CATEGORICAL_COLUMNS]

        for col in CATEGORICAL_COLUMNS:
            data[col] = data[col].replace({'None': 'No_Condition', 'none': 'No_Condition', np.nan: 'Unknown'})

        data[self.numerical_columns] = data[self.numerical_columns].apply(pd.to_numeric, errors='coerce')

        if fit:
            for col in CATEGORICAL_COLUMNS:
                categories = set(data[col].astype(str).unique())
                categories.add('Unknown')
                self.category_mappings[col] = categories
                encoder = LabelEncoder()
                encoder.fit(list(categories))
                self.label_encoders[col] = encoder
                data[col] = encoder.transform(data[col].astype(str))
        else:
            for col in CATEGORICAL_COLUMNS:
                data[col] = data[col].astype(str)
                mask = ~data[col].isin(self.category_mappings[col])
                data.loc[mask, col] = 'Unknown'
                data[col] = self.label_encoders[col].transform(data[col])

        self.feature_columns = self.numerical_columns + CATEGORICAL_COLUMNS
        X = data[self.feature_columns].astype(float).values
        numeric_idx = list(range(len(self.numerical_columns)))

        if fit:
            X[:, numeric_idx] = self.numeric_imputer.fit_transform(X[:, numeric_idx])
            X = self.scaler.fit_transform(X)
            self.fitted = True
        else:
            X[:, numeric_idx] = self.numeric_imputer.transform(X[:, numeric_idx])
            X = self.scaler.transform(X)

        return X
