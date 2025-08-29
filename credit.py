# Import necessary libraries
import pandas as pd  # Data manipulation and analysis
import numpy as np   # Numerical operations
from sklearn.model_selection import train_test_split  # Data splitting
from sklearn.linear_model import LogisticRegression   # ML model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # Evaluation metrics
from sklearn.preprocessing import StandardScaler  # Feature scaling

# Load the dataset
data = pd.read_csv('creditcard.csv')

# Display basic info and check missing values
print(data.head())
print(data.tail())
print(data.info())
print(data.isnull().sum())  # Should be zero

# Check distribution of 'Class' variable
print(data['Class'].value_counts())

# Separate legit and fraud transactions
legit = data[data.Class == 0]
fraud = data[data.Class == 1]

print('Legit transactions:', legit.shape)
print('Fraud transactions:', fraud.shape)

# Statistical overview of transaction amounts
print(legit.Amount.describe())
print(fraud.Amount.describe())

# Compare mean values grouped by Class
print(data.groupby('Class').mean())

# Undersample legit transactions to balance dataset
legit_sample = legit.sample(n=fraud.shape[0])
new_dataset = pd.concat([legit_sample, fraud], axis=0)

print(new_dataset['Class'].value_counts())
print(new_dataset.groupby('Class').mean())

# Define features and target
X = new_dataset.drop(columns='Class')
Y = new_dataset['Class']

print(X.shape, Y.shape)

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train and test sets with stratification
X_train, X_test, Y_train, Y_test = train_test_split(
    X_scaled, Y, test_size=0.2, stratify=Y, random_state=2
)

print('Train shape:', X_train.shape)
print('Test shape:', X_test.shape)

# Train Logistic Regression model with increased max iterations
model = LogisticRegression(max_iter=500)
model.fit(X_train, Y_train)

# Predictions on training and test data
Y_train_pred = model.predict(X_train)
Y_test_pred = model.predict(X_test)

# Evaluate model
print("Training Accuracy:", accuracy_score(Y_train, Y_train_pred))
print("Test Accuracy:", accuracy_score(Y_test, Y_test_pred))
print("Confusion Matrix:\n", confusion_matrix(Y_test, Y_test_pred))
print("Classification Report:\n", classification_report(Y_test, Y_test_pred))

# Make sure this is after model and scaler are trained, and X is your features DataFrame

feature_names = X.columns.tolist()  # Get feature names from X

print("Please enter values for the features to predict whether the transaction is fraud:")

user_input = []
for feature in feature_names:
    value = float(input(f"Enter value for {feature}: "))
    user_input.append(value)

# Scale the user input
user_input_scaled = scaler.transform([user_input])

# Predict using the trained Logistic Regression model
predicted_class = model.predict(user_input_scaled)

# Interpret the prediction
if predicted_class[0] == 1:
    print("The predicted class is 1: This transaction is likely FRAUDULENT.")
else:
    print("The predicted class is 0: This transaction is likely LEGITIMATE.")
