import pickle
import pandas as pd
import numpy as np
from constants import NOMINAL_COLUMNS, DISCRETE_COLUMNS, CONTINUOUS_COLUMNS
from scipy import stats
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import mlflow
from mlflow.artifacts import download_artifacts
import json

class MLModel:
    def __init__(self, client):
        """
        Initialize the MLModel with the given MLflow client and 
        load the staging model if available.

        Parameters:
            client (MlflowClient): The MLflow client used to 
            interact with the MLflow registry.

        Attributes:
            model (object): The loaded model, or None if no model 
                is loaded.
            fill_values_nominal (dict): Dictionary of fill values 
                for nominal columns.
            fill_values_discrete (dict): Dictionary of fill values 
                for discrete columns.
            fill_values_continuous (dict): Dictionary of fill values 
                for continuous columns.
            min_max_scaler_dict (dict): Dictionary of MinMaxScaler objects 
                for continuous columns.
            onehot_encoders (dict): Dictionary of OneHotEncoder objects 
                for nominal columns.
        """
        self.client = client
        self.model = None
        self.fill_values_nominal = None
        self.fill_values_discrete = None
        self.fill_values_continuous = None
        self.min_max_scaler_dict = None
        self.onehot_encoders = None
        self.load_staging_model()

    def load_staging_model(self):
        """
        Load the latest model tagged with 'Staging' stage from MLflow 
        if available.
        
        If a model with the 'Staging' tag exists, it loads the model 
        and associated artifacts. Otherwise, prints a warning.

        Returns:
            None
        """
        try:
            latest_staging_model = None
            for model in self.client.search_registered_models():
                for latest_version in model.latest_versions:
                    if latest_version.current_stage == "Staging":
                        latest_staging_model = latest_version
                        break
                if latest_staging_model:
                    break
            
            if latest_staging_model:
                model_uri = latest_staging_model.source
                self.model = mlflow.sklearn.load_model(model_uri)
                print("Staging model loaded successfully.")
                
                # Load associated artifacts
                artifact_uri = latest_staging_model.source.rpartition('/')[0]
                self.load_artifacts(artifact_uri)
            else:
                print("No staging model found.")
                
        except Exception as e:
            print(f"Error loading model or artifacts: {e}")

    def load_artifacts(self, artifact_uri):
        """
        Load necessary artifacts (e.g., scalers, encoders) from the given 
        artifact URI.

        Parameters:
            artifact_uri (str): The URI of the artifact directory containing 
            necessary files.

        Returns:
            None
        """
        try:
            # Load nominal fill values
            nominal_path = download_artifacts(artifact_uri=f"""
                    {artifact_uri}/fill_values_nominal.json""")
            with open(nominal_path, 'r') as f:
                self.fill_values_nominal = json.load(f)

            # Load discrete fill values
            discrete_path = download_artifacts(artifact_uri=f"""
                    {artifact_uri}/fill_values_discrete.json""")
            with open(discrete_path, 'r') as f:
                self.fill_values_discrete = json.load(f)

            # Load continuous fill values
            continuous_path = download_artifacts(artifact_uri=f"""
                    {artifact_uri}/fill_values_continuous.json""")
            with open(continuous_path, 'r') as f:
                self.fill_values_continuous = json.load(f)

            # Load MinMaxScaler dictionary
            scaler_path = download_artifacts(artifact_uri=f"""
                    {artifact_uri}/min_max_scaler_dict.pkl""")
            with open(scaler_path, 'rb') as f:
                self.min_max_scaler_dict = pickle.load(f)

            # Load OneHotEncoders
            encoders_path = download_artifacts(artifact_uri=f"""
                    {artifact_uri}/onehot_encoders.pkl""")
            with open(encoders_path, 'rb') as f:
                self.onehot_encoders = pickle.load(f)

            print("Artifacts loaded successfully.")

        except Exception as e:
            print(f"Error loading artifacts: {e}")

    def train_and_save_model(self, df):
        """
        Train an XGBoost model on the provided DataFrame 
        and calculate accuracy metrics.

        Parameters:
            df (pd.DataFrame): The training data containing 
                features and the target column 'V24'.

        Returns:
            tuple: A tuple containing:
                - train_accuracy (float): Accuracy on the training set.
                - test_accuracy (float): Accuracy on the test set.
                - xgb (XGBClassifier): The trained XGBoost model.
        """
        y = df["V24"]
        X = df.drop(columns="V24")
        X_train, X_test, y_train, y_test = train_test_split(X, 
                                                            y, 
                                                            test_size=0.10, 
                                                            random_state=42)

        xgb = XGBClassifier(max_depth=4, n_estimators=10)
        xgb.fit(X_train, y_train)
        self.model = xgb

        train_accuracy, test_accuracy = self.get_accuracy(X_train, 
                                                          X_test, 
                                                          y_train, 
                                                          y_test)
        
        return train_accuracy, test_accuracy, xgb

    def get_accuracy(self, X_train, X_test, y_train, y_test):
        """
        Calculate the accuracy of the model on both the training 
        and test datasets.

        Parameters:
            X_train (pd.DataFrame): Features for the training set.
            X_test (pd.DataFrame): Features for the test set.
            y_train (pd.Series): Actual labels for the training set.
            y_test (pd.Series): Actual labels for the test set.

        Returns:
            tuple: A tuple containing:
                - train_accuracy (float): Accuracy on the training set.
                - test_accuracy (float): Accuracy on the test set.
        """
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)

        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)

        print("Train Accuracy: ", train_accuracy)
        print("Test Accuracy: ", test_accuracy)

        return train_accuracy, test_accuracy

    def preprocessing_pipeline(self, df):
        """
        Preprocess the data by handling missing values, creating features, 
        encoding, and normalizing.

        Parameters:
            df (pd.DataFrame): Raw input data to preprocess.

        Returns:
            pd.DataFrame: Processed data ready for model 
                training or inference.
        """
        df = df.replace('?', np.nan)
        list_column_names = ["V" + str(i) for i in range(1, 29)]
        df.columns = list_column_names
        df = df.drop(columns=['V3'])

        for col in CONTINUOUS_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Fill missing values
        self.fill_values_nominal = {col: df[col].mode()[0] 
                                    for col in NOMINAL_COLUMNS}
        self.fill_values_discrete = {col: df[col].median() 
                                     for col in DISCRETE_COLUMNS}
        self.fill_values_continuous = {col: df[col].mean(skipna=True) 
                                       for col in CONTINUOUS_COLUMNS}

        for col in NOMINAL_COLUMNS:
            df[col].fillna(self.fill_values_nominal[col], inplace=True)
        for col in DISCRETE_COLUMNS:
            df[col].fillna(self.fill_values_discrete[col], inplace=True)
        for col in CONTINUOUS_COLUMNS:
            df[col].fillna(self.fill_values_continuous[col], inplace=True)

        # Handle outliers using Z-score
        for col in CONTINUOUS_COLUMNS:
            df[col + '_zscore'] = stats.zscore(df[col])
            outlier_indices = df[abs(df[col + '_zscore']) > 3].index
            df.loc[outlier_indices, col] = df[col].mean()
            df.drop(columns=[col + '_zscore'], inplace=True)

        # OneHot encoding for categorical features
        self.onehot_encoders = {}
        for col in NOMINAL_COLUMNS:
            encoder = OneHotEncoder(sparse_output=False, 
                                    handle_unknown='ignore')
            new_data = encoder.fit_transform(df[[col]])
            new_df = pd.DataFrame(new_data, 
                                  columns=encoder.get_feature_names_out([col]))
            df = pd.concat([df, new_df], axis=1)
            self.onehot_encoders[col] = encoder
        df.drop(columns=NOMINAL_COLUMNS, inplace=True)

        # MinMax scaling for continuous features
        self.min_max_scaler_dict = {}
        for col in df.columns:
            scaler = MinMaxScaler()
            df[col] = scaler.fit_transform(df[[col]])
            self.min_max_scaler_dict[col] = scaler

        # Log artifacts to MLflow
        mlflow.log_dict(self.fill_values_nominal, 
                        "fill_values_nominal.json")
        mlflow.log_dict(self.fill_values_discrete, 
                        "fill_values_discrete.json")
        mlflow.log_dict(self.fill_values_continuous, 
                        "fill_values_continuous.json")

        # Serialize and log scalers and encoders
        with open("min_max_scaler_dict.pkl", "wb") as f:
            pickle.dump(self.min_max_scaler_dict, f)
        mlflow.log_artifact("min_max_scaler_dict.pkl")

        with open("onehot_encoders.pkl", "wb") as f:
            pickle.dump(self.onehot_encoders, f)
        mlflow.log_artifact("onehot_encoders.pkl")

        return df

    def preprocessing_pipeline_inference(self, sample_data):
        """
        Preprocess a single row for inference to match the 
        training data structure.

        Parameters:
            sample_data (list): A list of values representing 
            a single row of data.

        Returns:
            pd.DataFrame: Processed data row ready for inference.
        """
        sample_data = [np.nan if item in ['?', 'null', None] 
                       else item for item in sample_data]
        sample_data = pd.DataFrame([sample_data], 
                                   columns=["V" + str(i) for i in range(1, 29)])
        sample_data = sample_data.replace(['?', 'null', 'None'], np.nan)
        sample_data = sample_data.drop(columns=['V3', 'V24'], errors='ignore')

        # Fill missing values using loaded artifacts
        for col in NOMINAL_COLUMNS:
            sample_data[col].fillna(self.fill_values_nominal[col], 
                                    inplace=True)
        for col in DISCRETE_COLUMNS:
            sample_data[col].fillna(self.fill_values_discrete[col], 
                                    inplace=True)
        for col in CONTINUOUS_COLUMNS:
            sample_data[col].fillna(self.fill_values_continuous[col], 
                                    inplace=True)

        # Apply one-hot encoding and scaling
        for col, encoder in self.onehot_encoders.items():
            new_data = encoder.transform(sample_data[[col]])
            new_df = pd.DataFrame(new_data, 
                                  columns=encoder.get_feature_names_out([col]))
            sample_data = pd.concat([sample_data, new_df], 
                                    axis=1).drop(columns=[col])
        
        for col, scaler in self.min_max_scaler_dict.items():
            if col in sample_data.columns:
                sample_data[col] = scaler.transform(sample_data[[col]])

        return sample_data

    def predict(self, inference_row):
        """
        Make a prediction using the preloaded staging model.

        Parameters:
            inference_row (list): A list of values representing a 
            single row of data for prediction.

        Returns:
            int: Predicted class label.
        """
        if self.model is None:
            return {'error': 'No staging model is loaded'}, 400

        processed_data = self.preprocessing_pipeline_inference(inference_row)
        prediction = self.model.predict(processed_data)
        return int(prediction)
    
    @staticmethod
    def create_new_folder(folder):
        """
        Create a new folder if it doesn't exist.

        Parameters:
            folder (str): Path to the folder.

        Returns:
            None
        """
        Path(folder).mkdir(parents=True, exist_ok=True)
 