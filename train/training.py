import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from skl2onnx import to_onnx
import bentoml


class TrainingPipeline():
    """Wrapper class for training the model"""

    def __call__(self, dataset_path):
        """Excecutes complete Training pipeline"""
        df = self.load_dataset(dataset_path)
        df = self.clean_data(df)

        X_train, X_test, y_train, y_test = self.split_data_for_model_training(df)

        model = self.train_model(X_train, y_train)

        self.validate_model_training(model, X_test, y_test)

        onnx_proto = self.convert_model_to_onnx(model, X_train[:1].values)

        self.save_model_to_bento(onnx_proto)

        print("Model Training Completed!")

    @staticmethod
    def load_dataset(dataset_path: str):
        """Load Dataset from a csv file"""
        df = pd.read_csv(dataset_path)
        return df
    
    @staticmethod
    def clean_data(df: pd.DataFrame):
        """Handling null values"""
        imputer = SimpleImputer(fill_value=0)
        df_imputed = df.copy()
        df_imputed[['Age', 'Annual_Income', 'Credit_Score', 'Loan_Amount', 'Number_of_Open_Accounts']] = imputer.fit_transform(df[['Age', 'Annual_Income', 'Credit_Score', 'Loan_Amount', 'Number_of_Open_Accounts']])
        return df_imputed
    
    @staticmethod
    def split_data_for_model_training(df_imputed:pd.DataFrame)-> tuple:
        """Split Data for model training"""
        # Splitting the data into features and target
        X = df_imputed.drop('Loan_Approval', axis=1)
        y = df_imputed['Loan_Approval']

        # Splitting the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)

        return X_train, X_test, y_train, y_test
    
    @staticmethod
    def train_model(X_train, y_train):
        """Training the Logistic Regression model"""
        model = LogisticRegression()
        model.fit(X_train, y_train)
        return model
    
    @staticmethod
    def validate_model_training(model, X_test, y_test):
        """Validate the model and save data for future validation"""
        y_pred = model.predict(X_test)

        report = classification_report(y_test, y_pred, output_dict=True)
        pd.DataFrame(report).transpose().to_csv("train/outputs/classification_report.csv")

        val_df = pd.concat([X_test, y_test], axis=1)
        val_df.to_csv("train/outputs/validation_data.csv", index=False)

    @staticmethod
    def convert_model_to_onnx(model, sample_arr:np.ndarray):
        """ Convert into ONNX format."""
        onnx_path = "train/outputs/lr_loans_1.onnx"
        onx = to_onnx(model, sample_arr, options={'zipmap': False})
        with open(onnx_path, "wb") as f:
            f.write(onx.SerializeToString())

        return onx

    @staticmethod
    def save_model_to_bento(model_proto: str):
        """Save model to Bento"""
        bento_model = bentoml.onnx.save_model("onnx_model", model_proto, signatures={"run": {"batchable": False}})
        print(f"Bento Model Tag: {bento_model.tag}")

if __name__ == "__main__":
    training_pipeline = TrainingPipeline()
    training_pipeline("train/dataset.csv")
