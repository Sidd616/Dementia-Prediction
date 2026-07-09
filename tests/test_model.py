import pandas as pd
import pytest

from model import INPUT_COLUMNS, DementiaPredictionModel

CATEGORY_CYCLES = {
    'Smoking_Status': ['Never Smoked', 'Former Smoker', 'Current Smoker'],
    'Physical_Activity': ['Sedentary', 'Mild Activity', 'Moderate Activity', 'High Activity'],
    'Nutrition_Diet': ['Balanced Diet', 'Low-Carb Diet', 'Mediterranean Diet', 'Other'],
    'Sleep_Quality': ['Good', 'Fair', 'Poor'],
    'Chronic_Health_Conditions': ['None', 'Heart Disease', 'Diabetes', 'Hypertension'],
}


def make_synthetic_dataset(n=40):
    rows = []
    for i in range(n):
        dementia = i % 2
        rows.append({
            'Diabetic': i % 2,
            'AlcoholLevel': round(0.01 * i, 4),
            'HeartRate': 60 + i,
            'BloodOxygenLevel': 95 + (i % 5),
            'BodyTemperature': round(36.0 + 0.01 * i, 4),
            'Weight': 55 + i,
            'MRI_Delay': 20 + i,
            'Prescription': 'Donepezil' if dementia else '',
            'Dosage in mg': 10 if dementia else '',
            'Age': 50 + i,
            'Dominant_Hand': 'Right' if i % 2 == 0 else 'Left',
            'Gender': 'Male' if i % 2 == 0 else 'Female',
            'Family_History': 'Yes' if i % 3 == 0 else 'No',
            'Smoking_Status': CATEGORY_CYCLES['Smoking_Status'][i % 3],
            'APOE_ε4': 'Positive' if i % 2 == 0 else 'Negative',
            'Physical_Activity': CATEGORY_CYCLES['Physical_Activity'][i % 4],
            'Depression_Status': 'Yes' if i % 4 == 0 else 'No',
            'Cognitive_Test_Scores': i % 11,
            'Medication_History': 'Yes' if dementia else 'No',
            'Nutrition_Diet': CATEGORY_CYCLES['Nutrition_Diet'][i % 4],
            'Sleep_Quality': CATEGORY_CYCLES['Sleep_Quality'][i % 3],
            'Chronic_Health_Conditions': CATEGORY_CYCLES['Chronic_Health_Conditions'][i % 4],
            'Dementia': dementia,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_csv(tmp_path):
    path = tmp_path / 'synthetic.csv'
    make_synthetic_dataset().to_csv(path, index=False)
    return str(path)


def test_train_and_predict_roundtrip(synthetic_csv):
    model = DementiaPredictionModel()
    model.train(synthetic_csv, test_size=0.25, cv_folds=3)
    assert model.is_trained

    row = make_synthetic_dataset(n=1).iloc[0]
    values = [str(row[col]) for col in INPUT_COLUMNS]
    predictions = model.predict(','.join(values))

    assert set(predictions.keys()) == {'Logistic Regression', 'Random Forest', 'XGBoost'}
    for probs in predictions.values():
        assert len(probs) == 2
        assert probs.sum() == pytest.approx(1.0, rel=1e-6)


def test_predict_before_train_raises():
    model = DementiaPredictionModel()
    with pytest.raises(RuntimeError):
        model.predict(','.join(['x'] * len(INPUT_COLUMNS)))


def test_predict_rejects_wrong_column_count(synthetic_csv):
    model = DementiaPredictionModel()
    model.train(synthetic_csv, test_size=0.25, cv_folds=3)
    with pytest.raises(ValueError):
        model.predict('only,a,few,values')


def test_save_and_load_roundtrip(tmp_path, synthetic_csv):
    model = DementiaPredictionModel()
    model.train(synthetic_csv, test_size=0.25, cv_folds=3)
    model_path = tmp_path / 'model.joblib'
    model.save(str(model_path))

    loaded = DementiaPredictionModel.load(str(model_path))
    row = make_synthetic_dataset(n=1).iloc[0]
    values = [str(row[col]) for col in INPUT_COLUMNS]
    predictions = loaded.predict(','.join(values))
    assert 'XGBoost' in predictions
