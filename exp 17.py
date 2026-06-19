# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# Create Mobile Price Dataset
data = {
    'RAM': [2, 4, 6, 8, 4, 8, 12, 16],
    'Storage': [16, 32, 64, 128, 64, 128, 256, 512],
    'Battery': [3000, 3500, 4000, 4500, 5000, 5000, 6000, 6000],
    'Camera': [8, 12, 16, 48, 32, 64, 108, 108],
    'Price_Range': [0, 1, 1, 2, 1, 2, 3, 3]
}


# Convert data into DataFrame
df = pd.DataFrame(data)


# Display dataset
print("Mobile Dataset:")
print(df)


# Features and Target
X = df[['RAM', 'Storage', 'Battery', 'Camera']]
y = df['Price_Range']


# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=1
)


# Create Random Forest Classifier
model = RandomForestClassifier(
    n_estimators=100,
    random_state=1
)


# Train model
model.fit(X_train, y_train)


# Predict test data
y_pred = model.predict(X_test)


# Evaluate model
print("\nMobile Price Prediction")
print("----------------------")

print("Accuracy:",
      accuracy_score(y_test, y_pred) * 100, "%")


print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Predict new mobile price
new_mobile = pd.DataFrame(
    [[8, 128, 5000, 64]],
    columns=['RAM', 'Storage', 'Battery', 'Camera']
)


prediction = model.predict(new_mobile)


print("\nNew Mobile Details:")
print(new_mobile)


# Display price category

if prediction[0] == 0:
    print("Predicted Price Range: Low Cost")

elif prediction[0] == 1:
    print("Predicted Price Range: Medium Cost")

elif prediction[0] == 2:
    print("Predicted Price Range: High Cost")

else:
    print("Predicted Price Range: Premium")
