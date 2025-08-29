# Credit Card Fraud Detection Project

## Project Overview

This project implements a machine learning model to detect fraudulent credit card transactions. Using anonymized transactional data, the goal is to classify transactions as legitimate or fraudulent based on feature patterns.

## Features

- Handles imbalanced dataset by undersampling legitimate transactions.
- Applies feature scaling for improved model convergence and accuracy.
- Utilizes Logistic Regression as a baseline classification model.
- Provides an interactive user input interface to test new transactions.
- Reports detailed evaluation metrics including precision, recall, and F1-score.

## Dataset

The project uses the Credit Card Fraud Detection dataset from Kaggle, which contains anonymized PCA-transformed features along with transaction amount and class labels indicating fraud status.

---

## Usage

1. Download and place the dataset as explained in the **Dataset Download Instructions** section.
2. Run the preprocessing and training scripts to build the model.
3. Use the interactive interface to input transaction features and predict fraud.

---

## Technology Stack

- Python (Pandas, NumPy)
- scikit-learn for machine learning and preprocessing
- Git and GitHub for version control

---

## Dataset Download Instructions

Due to GitHub file size limits, the dataset is not included in this repository.

Please download it separately from Kaggle:  
[Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  

Place the downloaded `creditcard.csv` in the root directory of this project before running the code.

---

## Contribution

Contributions and improvements are welcome. Please fork the repository and submit pull requests.

---

## License

Specify your project license here.

---

This README provides users with a clear understanding of the project’s goals, features, data, and setup guidelines.
