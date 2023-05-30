'''Unit tests for the model.py file.'''

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from src.ml.model import train_model, compute_model_metrics, inference


def test_train_model(test_data):
    '''
    Test the train_model function in the model.py file.
    '''
    x_train, y_train = test_data
    model = train_model(x_train, y_train)

    assert model is not None, "model is of type None"
    assert isinstance(model,RandomForestClassifier), "model is not of type RandomForestClassifier"

def test_compute_model_metrics(test_data):
    '''
    Test the compute_model_metrics function in the model.py file.
    '''

    x_train, y_train = test_data
    model = train_model(x_train, y_train)
    preds = inference(model, x_train)
    metrics = compute_model_metrics(y_train, preds)

    assert len(metrics) == 3
    assert isinstance(metrics, tuple)
    for metric in metrics:
        assert isinstance(metric, float)

def test_inference(test_data):
    '''
    Test the inference function in the model.py file.
    '''

    x_train, y_train = test_data
    model = train_model(x_train, y_train)
    preds = inference(model, x_train)
    assert len(preds) == len(x_train) 
    print(np.all((preds==0)|(preds == 1)))
    assert np.all((preds==0)|(preds == 1)) == True