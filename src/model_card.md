# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

The model used is a RandomForest classifier from ``sklearn.ensemble.RandomForestClassifier`` with default hyperparameters.

## Intended Use

The model was developed to predict if the annual salarary of a person may be more or less than $50k based on different features such as numeric ones (age, capital-gain,..) and categorial ones (education, workclass,...).



## Training Data

The training data is based on the Census Income Data Set: Predict whether income exceeds $50K/yr based on census data. Also known as "Adult" dataseset (https://archive.ics.uci.edu/ml/datasets/census+income).

| Data Set Characteristics  | Attribute Characteristics  | Associated Tasks | Number of Instances | Number of Attributes | Missing Values | Area | Date Donated | Number of Web Hits | 
|---|---|---|---|---|---|---|---|---|
| Multivariate | Categorical, Integer | Classification | 48842 | 14 | Yes | Social | 1996-05-01 | 775808 |

The original dataset contained trailing whitespaces that have been removed before training the actual model.

The cleaned dataset was split into 80\% for training and 20\% for testing. A one-hot encoder was used for categorial features, a label binarizer was used on the target variable and no stratification was done.

## Evaluation Data

As mentioned before, the original dataset was split into 80\% for training and 20\% for testing. Therefore, the evaluation data is 20\% of the original, cleaned dataset.

## Metrics

The following metrics were used to evaluate the Random Forest classifier on the given test set:

- **Precision**: Precision is a measure of the accuracy of a classifier when it predicts a positive result. It's defined as the number of true positive predictions (i.e., the number of items correctly labeled as belonging to the positive class) divided by the total number of positive predictions (i.e., the sum of true positive and false positive predictions). In other words, it answers the question "When the model predicts positive, how often is it correct?"

- **Recall**: Recall, also known as sensitivity or true positive rate, measures the ability of a classifier to identify all positive instances. It is calculated as the number of true positive predictions divided by the total number of actual positive instances (i.e., the sum of true positive and false negative predictions). So, recall answers the question "When it's actually positive, how often does the model predict positive?"

- F-Beta score: F-beta score is a measure that combines both precision and recall into a single number. The F-beta score is the weighted harmonic mean of precision and recall, with the beta parameter determining the weight of recall in the combined score. The beta parameter quantifies the tradeoff between precision and recall in the combined score. For example, when beta is 1, we have the F1 score, which gives equal weight to both precision and recall. When beta is less than 1, more emphasis is put on precision, and when beta is greater than 1, more weight is given to recall.

Actual metric values achieved by the model on the test set:

| Metric  | Value  |
|---|---|
| Precision | 0.7440061871616396 |
| Recall    | 0.6263020833333334 |
| F-Beta    | 0.6800989749027925 | 



## Ethical Considerations

There was no investigation done in terms of bias or if the dataset is unbalanced. The model fairness is therefore not guaranteed.


## Caveats and Recommendations

Data extraction was done by Barry Becker from the 1994 Census database. Therefore, the data might be outdated. Furtheremore, there seems to be bias when predicitions are done on slices of the data.