import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

class DementiaPredictionModel:
    def __init__(self):
        self.label_encoders = {}
        self.imputer = SimpleImputer(strategy='constant', fill_value=0)
        self.scaler = StandardScaler()
        self.lr_model = LogisticRegression(max_iter=1000)
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
        self.category_mappings = {}
        
    def preprocess_data(self, data, is_training=True):
        # Create a copy of the data to avoid modifying the original
        data = data.copy()
        
        # Define categorical and numerical columns
        categorical_columns = [
            'Prescription', 'Dominant_Hand', 'Gender', 'Family_History',
            'Smoking_Status', 'APOE_ε4', 'Physical_Activity', 'Depression_Status',
            'Medication_History', 'Nutrition_Diet', 'Sleep_Quality',
            'Chronic_Health_Conditions'
        ]
        
        numerical_columns = [col for col in data.columns if col not in categorical_columns]
        
        # Handle missing values and 'None' values in categorical columns
        for col in categorical_columns:
            data[col] = data[col].replace({'None': 'No_Condition', 'none': 'No_Condition', np.nan: 'Unknown'})
        
        # Handle missing values in numerical columns
        data[numerical_columns] = data[numerical_columns].apply(pd.to_numeric, errors='coerce')
        
        # Encode categorical variables
        if is_training:
            # Store unique categories for each categorical column
            for col in categorical_columns:
                unique_categories = set(data[col].astype(str).unique())
                unique_categories.add('Unknown')  # Add 'Unknown' category
                self.category_mappings[col] = list(unique_categories)
                
                self.label_encoders[col] = LabelEncoder()
                self.label_encoders[col].fit(list(unique_categories))
                data[col] = self.label_encoders[col].transform(data[col].astype(str))
        else:
            # Handle unseen categories in prediction data
            for col in categorical_columns:
                # Replace unseen categories with 'Unknown'
                data[col] = data[col].astype(str)
                mask = ~data[col].isin(self.category_mappings[col])
                data.loc[mask, col] = 'Unknown'
                data[col] = self.label_encoders[col].transform(data[col])
        
        # Convert all features to float
        X = data.astype(float)
        
        # Impute missing values
        if is_training:
            X = self.imputer.fit_transform(X)
        else:
            X = self.imputer.transform(X)
            
        # Scale features
        if is_training:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)
            
        return X

    def train(self, data_path):
        # Load and preprocess training data
        data = pd.read_csv(data_path, encoding='utf-8')
        
        # Separate features and target
        X = data.drop('Dementia', axis=1)
        y = data['Dementia']
        
        # Preprocess features
        X_processed = self.preprocess_data(X, is_training=True)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.2, random_state=42
        )
        
        # Train models
        self.lr_model.fit(X_train, y_train)
        self.rf_model.fit(X_train, y_train)
        self.xgb_model.fit(X_train, y_train)
        
        # Evaluate models
        models = {
            'Logistic Regression': self.lr_model,
            'Random Forest': self.rf_model,
            'XGBoost': self.xgb_model
        }
        
        for name, model in models.items():
            predictions = model.predict(X_test)
            print(f"\n{name} Results:")
            print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
            print("Classification Report:")
            print(classification_report(y_test, predictions))

    def predict(self, input_data_str):
        try:
            # Convert input string to DataFrame
            columns = [
                'Diabetic', 'AlcoholLevel', 'HeartRate', 'BloodOxygenLevel', 
                'BodyTemperature', 'Weight', 'MRI_Delay', 'Prescription', 
                'Dosage in mg', 'Age', 'Dominant_Hand', 'Gender', 
                'Family_History', 'Smoking_Status', 'APOE_ε4', 
                'Physical_Activity', 'Depression_Status', 'Cognitive_Test_Scores', 
                'Medication_History', 'Nutrition_Diet', 'Sleep_Quality', 
                'Chronic_Health_Conditions'
            ]
            
            input_data = pd.DataFrame([input_data_str.split(',')], columns=columns)
            
            # Preprocess input data
            X_processed = self.preprocess_data(input_data, is_training=False)
            
            # Make predictions
            predictions = {
                'Logistic Regression': self.lr_model.predict_proba(X_processed)[0],
                'Random Forest': self.rf_model.predict_proba(X_processed)[0],
                'XGBoost': self.xgb_model.predict_proba(X_processed)[0]
            }
            
            return predictions
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return None

# Usage example
def main():
    try:
        # Initialize and train the model
        model = DementiaPredictionModel()
        model.train('data.csv')
        
        # Example input data
        input_data = "0,0.000955737,84,99.84323059,36.03250039,84.81595461,38.72863817,,,49,Right,Female,No,Never Smoked,Negative,Mild Activity,No,10,No,Low-Carb Diet,Good,None"
        
        # Get predictions
        predictions = model.predict(input_data)
        
        if predictions:
            # Print predictions with probability scores
            print("\nPredictions for input data:")
            for model_name, probs in predictions.items():
                print(f"\n{model_name}:")
                print(f"Probability of No Dementia: {probs[0]:.4f}")
                print(f"Probability of Dementia: {probs[1]:.4f}")
        else:
            print("Prediction failed. Please check the input data format and try again.")
            
    except Exception as e:
        print(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    main()