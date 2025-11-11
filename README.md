# Iris Classification

This project implements an Iris flower classification model using machine learning algorithms.

## Description

The script uses the famous Iris dataset to train Decision Tree and Logistic Regression models for classifying Iris species based on sepal and petal measurements. It evaluates the models and allows user input for predictions.

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## How to Run

1. Ensure you have the required libraries installed:
   ```
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
2. Run the script:
   ```
   python .py
   ```
3. Enter the sepal length, sepal width, petal length, and petal width when prompted to get a species prediction.

## Files

- `.py`: Main classification script
- `IRIS.csv`: Iris dataset
- `README.md`: This file

## Output

The script will display:
- Dataset preview
- Model accuracies and classification reports
- Confusion matrix heatmap
- Predicted species for user input