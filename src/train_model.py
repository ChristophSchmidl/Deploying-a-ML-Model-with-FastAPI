# Script to train machine learning model.
from sklearn.model_selection import train_test_split
import logging
import pickle
import os
# Add the necessary imports for the starter code.
from ml.data import process_data, slice_data
from ml.model import train_model, compute_model_metrics, inference
import pandas as pd

if __name__ == "__main__":

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

    # Add code to load in the data.
    data_filepath = os.path.join(current_dir, "data", 'census_clean.csv') 
    data = pd.read_csv(data_filepath)

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
    pickle.dump(model, open(os.path.join(current_dir, "model", "model.pkl"), 'wb'))
    pickle.dump(encoder, open(os.path.join(current_dir, "model", "encoder.pkl"), 'wb'))
    pickle.dump(lb, open(os.path.join(current_dir, "model", "lb.pkl"), 'wb'))

    logger.info('Inference on test dataset...')
    preds = inference(model, X_test)

    logger.info('Compute model metrics...')
    precision, recall, fbeta = compute_model_metrics(y_test, preds)

    logger.info(f"Precision: {precision}")
    logger.info(f"Recall: {recall}")
    logger.info(f"FBeta: {fbeta}")


    # compute models metrics on the slices of the data for categorical variables and save them in slice_output.txt
    with open(os.path.join(current_dir, "model", "slice_metrics.txt"), 'w') as file:
        logging.info("Computing slices...")
        for feature in cat_features:
            df_result = slice_data(test, feature, y_test, preds)
            df_string = df_result.to_string(header=False, col_space=[20, 20, 20, 20, 20], index=False)
            file.write(df_string + "\n")