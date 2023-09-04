import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

class AdvancedRiskProfiler:

    def __init__(self, historical_data_path):
        self.data = pd.read_csv(historical_data_path)
        self.model = None
        self._prepare_data()
        self._train_model()

    def _prepare_data(self):
        # Example: Convert categorical data into numerical
        label_encoders = {}
        for column in ['medical_history', 'job_type', 'location']:
            le = LabelEncoder()
            self.data[column] = le.fit_transform(self.data[column])
            label_encoders[column] = le

        self.label_encoders = label_encoders

    def _train_model(self):
        # Splitting data into training and testing sets
        X = self.data.drop("risk_profile", axis=1)
        y = self.data["risk_profile"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Using a decision tree classifier for risk profiling
        clf = DecisionTreeClassifier(max_depth=5)
        clf.fit(X_train, y_train)

        # Store the trained model
        self.model = clf

    def predict_risk(self, client_data):
        # Convert client_data into a DataFrame for prediction
        df = pd.DataFrame([client_data])

        # Transform the data using label encoders
        for column, encoder in self.label_encoders.items():
            df[column] = encoder.transform(df[column])

        risk_prediction = self.model.predict(df)
        return risk_prediction[0]
