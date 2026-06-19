# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report


# Create Bank Loan Dataset
data = {
    'Age': [25, 35, 45, 30, 50, 40, 28, 60],
    'Income': [30000, 60000, 80000, 45000, 90000, 75000, 40000, 100000],
    'CreditScore': [600, 750, 800, 650, 850, 780, 620, 900],
    'LoanAmount': [200000, 300000, 500000, 250000, 600000, 450000, 150000, 700000],
    'Loan_Status': [0, 1, 1, 0, 1, 1, 0, 1]
}


# Convert data into DataFrame
df = pd.DataFrame(data)


# Display dataset
print("Bank Loan Dataset:")
print(df)


# Features and target
X = df[['Age', 'Income', 'CreditScore', 'LoanAmount']]
y = df['Loan_Status']


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=1
)


# Create Naive Bayes model
model = GaussianNB()


# Train model
model.fit(X_train, y_train)


# Prediction
y_pred = model.predict(X_test)


# Evaluation
print("\nBank Loan Prediction")
print("--------------------")

print("Accuracy:",
      accuracy_score(y_test, y_pred) * 100, "%")


print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Predict new customer loan status
new_customer = pd.DataFrame(
    [[35, 60000, 750, 300000]],
    columns=['Age','Income','CreditScore','LoanAmount']
)


prediction = model.predict(new_customer)


print("\nNew Customer Details:")
print(new_customer)


if prediction[0] == 1:
    print("Predicted Result: Loan Approved")
else:
    print("Predicted Result: Loan Rejected")
