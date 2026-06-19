# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Create sample house dataset
data = {
    'Area': [1000, 1200, 1500, 1800, 2000, 2500, 3000],
    'Bedrooms': [2, 2, 3, 3, 4, 4, 5],
    'Bathrooms': [1, 2, 2, 3, 3, 4, 4],
    'Age': [20, 15, 10, 8, 5, 3, 1],
    'Price': [2500000, 3200000, 4500000, 5500000, 
              7000000, 8500000, 10000000]
}

# Convert into DataFrame
df = pd.DataFrame(data)

# Input features and output
X = df[['Area', 'Bedrooms', 'Bathrooms', 'Age']]
y = df['Price']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

# Create Linear Regression model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("House Price Prediction")
print("----------------------")
print("Mean Absolute Error:",
      mean_absolute_error(y_test, y_pred))

print("R2 Score:",
      r2_score(y_test, y_pred))


# Predict new house price
new_house = [[2200, 4, 3, 5]]

price = model.predict(new_house)

print("\nNew House Details:")
print("Area = 2200 sq.ft")
print("Bedrooms = 4")
print("Bathrooms = 3")
print("Age = 5 years")

print("Predicted House Price: ₹", int(price[0]))
