import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from xgboost import XGBClassifier

from preprocessing import DementiaPreprocessor

RANDOM_STATE = 42
DEFAULT_MODEL_PATH = 'dementia_model.joblib'

# Columns expected in a raw prediction request, in order. Deliberately
# excludes Prescription / Dosage in mg / Medication_History: those fields are
# only known once a patient is already diagnosed and on medication, so they
# can't be used to predict an undiagnosed patient's status (see
# preprocessing.LEAKAGE_COLUMNS).
INPUT_COLUMNS = [
    'Diabetic', 'AlcoholLevel', 'HeartRate', 'BloodOxygenLevel',
    'BodyTemperature', 'Weight', 'MRI_Delay', 'Age', 'Dominant_Hand', 'Gender',
    'Family_History', 'Smoking_Status', 'APOE_ε4',
    'Physical_Activity', 'Depression_Status', 'Cognitive_Test_Scores',
    'Nutrition_Diet', 'Sleep_Quality', 'Chronic_Health_Conditions'
]


class DementiaPredictionModel:
    def __init__(self):
        self.preprocessor = DementiaPreprocessor()
        self.lr_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
        self.xgb_model = XGBClassifier(eval_metric='logloss', random_state=RANDOM_STATE)
        self.models = {
            'Logistic Regression': self.lr_model,
            'Random Forest': self.rf_model,
            'XGBoost': self.xgb_model,
        }
        self.is_trained = False

    def train(self, data_path, test_size=0.2, cv_folds=5):
        data = pd.read_csv(data_path, encoding='utf-8')

        X = data.drop('Dementia', axis=1)
        y = data['Dementia']

        # Split BEFORE fitting any preprocessing so the imputer, scaler, and
        # label encoders never see held-out rows.
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
        )

        X_train = self.preprocessor.fit_transform(X_train_raw)
        X_test = self.preprocessor.transform(X_test_raw)

        for name, model in self.models.items():
            model.fit(X_train, y_train)

            cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds)
            predictions = model.predict(X_test)

            print(f"\n{name} Results:")
            print(f"Cross-val accuracy ({cv_folds}-fold, train split): {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
            print(f"Held-out test accuracy: {accuracy_score(y_test, predictions):.4f}")
            print("Classification Report:")
            print(classification_report(y_test, predictions))

        self.is_trained = True

    def predict(self, input_data_str):
        if not self.is_trained:
            raise RuntimeError("Model must be trained (or loaded) before calling predict().")

        values = input_data_str.split(',')
        if len(values) != len(INPUT_COLUMNS):
            raise ValueError(
                f"Expected {len(INPUT_COLUMNS)} comma-separated values "
                f"({', '.join(INPUT_COLUMNS)}), got {len(values)}."
            )

        input_data = pd.DataFrame([values], columns=INPUT_COLUMNS)
        X_processed = self.preprocessor.transform(input_data)

        return {
            name: model.predict_proba(X_processed)[0]
            for name, model in self.models.items()
        }

    def save(self, path=DEFAULT_MODEL_PATH):
        if not self.is_trained:
            raise RuntimeError("Cannot save an untrained model.")
        joblib.dump(self, path)

    @staticmethod
    def load(path=DEFAULT_MODEL_PATH):
        return joblib.load(path)


def load_or_train_model(data_path='data.csv', model_path=DEFAULT_MODEL_PATH):
    """Load a cached trained model from disk, or train a fresh one and cache it."""
    if os.path.exists(model_path):
        print(f"Loading cached model from {model_path}")
        return DementiaPredictionModel.load(model_path)

    print("No cached model found, training a new one...")
    model = DementiaPredictionModel()
    model.train(data_path)
    model.save(model_path)
    return model


def main():
    model = load_or_train_model()

    # Example input, in INPUT_COLUMNS order (no Prescription/Dosage/Medication_History).
    input_data = "0,0.000955737,84,99.84323059,36.03250039,84.81595461,38.72863817,49,Right,Female,No,Never Smoked,Negative,Mild Activity,No,10,Low-Carb Diet,Good,None"

    predictions = model.predict(input_data)

    print("\nPredictions for input data:")
    for model_name, probs in predictions.items():
        print(f"\n{model_name}:")
        print(f"Probability of No Dementia: {probs[0]:.4f}")
        print(f"Probability of Dementia: {probs[1]:.4f}")


if __name__ == "__main__":
    main()
