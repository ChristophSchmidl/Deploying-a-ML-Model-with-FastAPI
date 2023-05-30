# Script to train machine learning model.
from sklearn.model_selection import train_test_split
import logging
import pickle
import os
# Add the necessary imports for the starter code.
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference
import pandas as pd


if not os.path.exists(os.path.join(os.getcwd(), "logs")):    
    os.makedirs(os.path.join(os.getcwd(), "logs"))

logging.basicConfig(
        filename='./logs/application.log',
        level=logging.INFO,
        filemode='a', # w =  overwrite, a = append
        format='%(asctime)s %(levelname)-8s %(filename)s:%(lineno)d %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
)    

logger = logging.getLogger()

# Add code to load in the data.
data = pd.read_csv("data/census_clean.csv")

# Optional enhancement, use K-fold cross validation instead of a train-test split.
logger.info('Splitting data into train and test set...')
train, test = train_test_split(data, test_size=0.20)

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


# Proces the test data with the process_data function.
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False,
    encoder = encoder, lb = lb
)


# Train and save a model.
logger.info('Training the model...')
model = train_model(X_train, y_train)

logger.info('Saving the model...')
pickle.dump(model, open(os.path.join("model", "model.pkl"), 'wb'))
pickle.dump(encoder, open(os.path.join("model", "encoder.pkl"), 'wb'))
pickle.dump(lb, open(os.path.join("model", "lb.pkl"), 'wb'))

logger.info('Inference on test dataset...')
preds = inference(model, X_test)

logger.info('Compute model metrics...')
precision, recall, fbeta = compute_model_metrics(y_test, preds)

logger.info(f"Precision: {precision}")
logger.info(f"Recall: {recall}")
logger.info(f"FBeta: {fbeta}")