import requests
import json

URL = "https://fastapi-service.onrender.com/predict"


# explicit the sample to perform inference on
sample =  {
            'age': 50,
            'workclass':"Private", 
            'fnlgt':234721,
            'education':"Doctorate",
            'education-num': 16,
            'marital-status':"Separated",
            'occupation':"Exec-managerial",
            'relationship':"Not-in-family",
            'race':"Black",
            'sex':"Female",
            'capital-gain':0,
            'capital-loss':0,
            'hours-per-week':50,
            'native-country':"United-States"
}


# post to API and collect response
response = requests.post(URL, json=sample)

# display output - response will show sample details + model prediction added
print("response status code", response.status_code)
print("response content:")
print(response.content)

'''
(fastapi-deployment) ➜  Deploying-a-ML-Model-with-FastAPI git:(main) ✗ python src/post_to_render.py
response status code 200
response content:
b'{"prediction":">50K"}'
'''