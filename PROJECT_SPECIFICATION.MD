# Project specification

## git

| Criteria  | Meets specifications  |
|---|---|
| Set up git with GitHub Actions. | - GitHub action should run pytest and flake8 on push to main/master.<br>- PyTest must pass (by time the project is done there should be at least six tests) and flake8 must pass without errors.<br>- Include either a link to your GitHub repository or a screenshot of the CI passing called continuous_integration.png. |

## Model building

| Criteria  | Meets specifications  |
|---|---|
| Create a machine learning model. | -The model should train on the provided data. The data should either be split to have a train-test split or use cross-validation on the entire dataset.<br>-Implement all stubbed functions in the starter code or create equivalents. At a minimum, there should be functions to:<br><br>1. train, save and load the model and any categorical encoders<br>2. model inference<br>3. determine the classification metrics.<br><br>- Write a script that takes in the data, processes it, trains the model, and saves it and the encoder. This script must use the functions you have written.|
| Write unit tests. | Write at least 3 unit tests. Unit testing ML can be hard due to the stochasticity -- at least test if any ML functions return the expected type. |
| Write a function that computes model metrics on slices of the data. | - Write a function that computes performance on model slices. I.e. a function that computes the performance metrics when the value of a given feature is held fixed. E.g. for education, it would print out the model metrics for each slice of data that has a particular value for education. You should have one set of outputs for every single unique value in education.<br>- Complete the stubbed function or write a new one that for a given categorical variable computes the metrics when its value is held fixed.<br>- Write a script that runs this function (or include it as part of the training script) that iterates through the distinct values in one of the features and prints out the model metrics for each value.<br>- Output the printout to a file named slice_output.txt. |
| Write a model card. | - The model card should address every section of the template. Please use the provided Model Card Template.<br>- The model card should be written in complete sentences and include metrics on model performance. Please include both the metrics used and your model's performance on those metrics. |

## API Creation

| Criteria  | Meets specifications  |
|---|---|
| Create a REST API. | -The API must implement GET and POST. GET must be on the root domain and give a greeting and POST on a different path that does model inference.<br>- Use Python type hints such that FastAPI creates the automatic documentation.<br>- Use a Pydantic model to ingest the body of the POST. This should implement an example (hint: Pydantic/FastAPI provides multiple ways to do this, see the docs for more information: https://fastapi.tiangolo.com/tutorial/schema-extra-example/).<br>- Include a screenshot of the docs that shows the example and name it example.png. |
| Create tests for an API. | You should write at least three test cases:<br>- A test case for the GET method. This MUST test both the status code as well as the contents of the request object.<br>- One test case for EACH of the possible inferences (results/outputs) of the ML model. |

## API Deployment

| Criteria  | Meets specifications  |
|---|---|
| Deploy an app to a Cloud Application Platform | - Deploy using a GitHub repository with Continuous Delivery enabled. Include a screenshot showing that CD is enabled and label it continuous_deloyment.png.<br>- Include a screenshot of your browser receiving the contents of the GET you implemented on the root domain. Name this screenshot live_get.png. |
| Query live API. | Write a script that POSTS to the API using the requests module and returns both the result of model inference and the status code. Include a screenshot of the result. Name this live_post.png. |

## Suggestions to make your project stand out!

1. Use codecov or equivalent to measure the test coverage (then write more tests and try to hit 100% coverage!).
2. Implement FastAPI’s advanced features such as authentication.
3. Create a front end for the API.