'''Test API endpoints.'''

import json
from fastapi.testclient import TestClient

from src.main import app

def test_root():
    '''Test the root endpoint.'''
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == "The API is working :)"

def test_predict_lowsalary(api_data_lowsalary):
    '''Test the predict endpoint for label low salary'''
    data = json.dumps(api_data_lowsalary)
    with TestClient(app) as client:
        response = client.post("/predict", data = data)
        assert response.status_code == 200
        assert response.json() == {"prediction": "<=50K"}

def test_predict_highsalary(api_data_highsalary):
    '''Test the predict endpoint for label high salary'''
    data = json.dumps(api_data_highsalary)
    with TestClient(app) as client:
        response = client.post("/predict", data = data)
        assert response.status_code == 200
        assert response.json() == {"prediction": ">50K"}