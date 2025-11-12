import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ml.model import train_model, compute_model_metrics, inference
from ml.data import process_data
# TODO: add necessary import

@pytest.fixture
def sample_data():
    # A minimal DataFrame similar to the census dataset
    df = pd.DataFrame({
        "age": [25, 45, 30, 50],
        "workclass": ["Private", "Self-emp", "Private", "Government"],
        "education": ["Bachelors", "Masters", "HS-grad", "Doctorate"],
        "marital-status": ["Never-married", "Married-civ-spouse", "Divorced", "Married-civ-spouse"],
        "occupation": ["Tech-support", "Exec-managerial", "Sales", "Prof-specialty"],
        "relationship": ["Not-in-family", "Husband", "Unmarried", "Husband"],
        "race": ["White", "Black", "White", "Asian-Pac-Islander"],
        "sex": ["Male", "Female", "Female", "Male"],
        "native-country": ["United-States", "United-States", "Canada", "India"],
        "salary": [">50K", "<=50K", "<=50K", ">50K"]
    })
    return df


@pytest.fixture
def processed_data(sample_data):
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
    X, y, encoder, lb = process_data(
        sample_data,
        categorical_features=cat_features,
        label="salary",
        training=True,
    )
    return X, y, encoder, lb


# check that the algorithm used to train is Random Forest Classifier
def test_used_algorithm(processed_data):
    X_train, y_train, _, _ = processed_data
    model = train_model(X_train, y_train)

    # either check by class name...
    expected_algorithm = "RandomForestClassifier"
    assert type(model).__name__ == expected_algorithm


# check that the inference model is returning predictions of the correct shape
def test_inference(processed_data):
    X, y, _, _ = processed_data
    model = train_model(X, y)
    preds = inference(model, X)
    assert preds.shape == y.shape


# check the compute_model_metrics returns float valies for precision, recall and fbeta
def test_compute_model_metrics_returns_floats():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])

    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)

    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)
