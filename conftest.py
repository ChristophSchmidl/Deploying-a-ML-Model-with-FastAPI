'''Pytest fixtures for testing.'''
import pytest
import pandas as pd
from sklearn.model_selection import train_test_split
from src.ml.data import process_data
import os

current_dir = os.path.dirname(os.path.realpath(__file__))


############################################
#              pytest fixtures
############################################

@pytest.fixture(scope="session")
def test_data():
    """
    Prepare data for machine learning model.
    """

    filepath = os.path.join(current_dir, "src", "data", 'census_clean.csv') 
    df = pd.read_csv(filepath)
    train, _ = train_test_split(df, test_size=0.20)

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    X_train, y_train, _, _ = process_data(
    train, categorical_features=cat_features, label="salary", training=True
        )

    return X_train, y_train

@pytest.fixture(scope="session")
def api_data_lowsalary():
    """Prepare data low salary for testing API model inference."""
    data_dict = {
        "age": 39,
        "workclass": "State-gov",
        "fnlwgt": 77516,
        "education": "Bachelors",
        "education-num" : 13,
        "marital-status" : "Never-married",
        "occupation" : "Adm-clerical",
        "relationship" : "Not-in-family",
        "race" : "White",
        "sex" : "Male",
        "capital-gain" : 2174,
        "capital-loss" : 0,
        "hours-per-week" : 40,
        "native-country"  : "United-States"

    }
    return data_dict

@pytest.fixture(scope="session")
def api_data_highsalary():
    """Prepare data high salary for testing API model inference."""
    return {
        "age": 52,
        "workclass": "Self-emp-inc",
        "fnlwgt": 209642,
        "education": "HS-grad",
        "education-num" : 9,
        "marital-status" : "Married-civ-spouse",
        "occupation" : "Exec-managerial",
        "relationship" : "Wife",
        "race" : "White",
        "sex" : "Female",
        "capital-gain" : 15024,
        "capital-loss" : 0,
        "hours-per-week" : 40,
        "native-country"  : "United-States"
    }