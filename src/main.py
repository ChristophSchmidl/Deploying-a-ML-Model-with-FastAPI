# Put the code for your API here.
from typing import Union
import pickle
import os
import logging
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from .ml.data import process_data
from .ml.model import inference


current_dir = os.path.dirname(os.path.realpath(__file__))

if not os.path.exists(os.path.join(current_dir, "logs")):    
    os.makedirs(os.path.join(current_dir, "logs"))

logging.basicConfig(
        filename='src/logs/application.log',
        level=logging.INFO,
        filemode='a', # w =  overwrite, a = append
        format='%(asctime)s %(levelname)-8s %(filename)s:%(lineno)d %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
)    

logger = logging.getLogger()



class Input(BaseModel):
    """Input data model"""
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num : int = Field(None, alias="education-num")
    marital_status : str = Field(None,  alias="marital-status")
    occupation : str
    relationship : str
    race : str
    sex : str
    capital_gain : int = Field(None, alias="capital-gain")
    capital_loss : int =  Field(None, alias="capital-loss")
    hours_per_week : int = Field(None, alias="hours-per-week")
    native_country  : str = Field(None, alias="native-country")

    # See https://fastapi.tiangolo.com/tutorial/schema-extra-example/
    # The example is present in the docs later on
    class Config:
        schema_extra = {
                        "example": {
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
                        }

class Prediction(BaseModel):
    """Prediction data model"""

    prediction: str

"""
Starting the app: uvicorn --host 0.0.0.0 --port 5000 --workers 4 src.main:app
"""
app = FastAPI()

@app.on_event("startup")
async def startup_event():
    global MODEL, ENCODER, BINARIZER

    with open(os.path.join(current_dir, "model", "model.pkl"), "rb") as f:
        MODEL = pickle.load(f)

    with open(os.path.join(current_dir, "model", "encoder.pkl"), "rb") as f:
        ENCODER = pickle.load(f)

    with open(os.path.join(current_dir, "model", "lb.pkl"), "rb") as f:
        BINARIZER = pickle.load(f)

@app.get("/")
async def welcome():
    """welcome message
    Returns:
        str: Welcome message
    """ 
    return "The API is working :)"

@app.post("/predict", response_model=Prediction, status_code=200)
async def predict(data_input: Input):
    """Performs an inference on a trained model with input data.

    Args:
        data (pd.DataFrame): Input data to the model.

    Returns:
        dict: The model predictions.
    """
    
    #logger.info(data_input)

    # dirty hack to match the names of the features with the expected names
    # for the pre-processing.
    data = {'age': data_input.age,
            'workclass': data_input.workclass, 
            'fnlgt': data_input.fnlgt,
            'education': data_input.education,
            'education-num': data_input.education_num,
            'marital-status': data_input.marital_status,
            'occupation': data_input.occupation,
            'relationship': data_input.relationship,
            'race': data_input.race,
            'sex': data_input.sex,
            'capital-gain': data_input.capital_gain,
            'capital-loss': data_input.capital_loss,
            'hours-per-week': data_input.hours_per_week,
            'native-country': data_input.native_country,
    }

    # prepare the sample for inference as a dataframe
    data_df = pd.DataFrame([data])

    #logger.info(data_df)

    # Convert the input data into a dataframe.
    #data_df = pd.DataFrame([data.dict()])

    #print(data_df)

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

    #data_df.fillna(data_df.mean(), inplace=True)

    # Preprocess the input data.
    X, _, _, _ = process_data(
        data_df, 
        categorical_features=cat_features, 
        training=False,
        encoder=ENCODER, 
        lb=BINARIZER
    )

    

    # Get the model's predictions.
    prediction = inference(MODEL, X)

    if prediction == 0:
        prediction = "<=50K"
    else:
        prediction = ">50K"

    return {"prediction": prediction}
