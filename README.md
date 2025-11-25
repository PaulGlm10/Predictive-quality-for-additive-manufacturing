# PredictiveLearningProject

## Dataset

The dataset contains feature-extracted samples where:

- Each row represents one sample
- Each sample has approximately 100 features
- Data should be split by row or by filename (e.g., 169.csv, 170.csv, etc.)
- **Column 2 (Label)**: Contains the data label which is very important
  - `ok (iO)`: Samples that are functioning correctly
  - `not ok (niO)`: Samples with errors
- **Column 3 (Error_Description)**: Contains the error description for the not ok samples

## Classification Task

The project aims to perform **multi-class classification**, classifying samples into 4 classes:

- **ok**: Functioning correctly
- **error type oil**: Oil-related error
- **error type tilt**: Tilt-related error
- **error type offset**: Offset-related error

## Feature Selection

A forward selection wrapper method will be implemented to find the best combination of features for each algorithm. The best feature combination for one algorithm may not be optimal for another algorithm, so forward selection must be performed separately for each algorithm.

## Model Evaluation

All models should be evaluated on the following metrics:

- **Accuracy**: Overall correctness of predictions
- **F1 Score**: Harmonic mean of precision and recall
- **Precision**: Proportion of positive predictions that are correct
- **Recall**: Proportion of actual positives that are correctly identified
