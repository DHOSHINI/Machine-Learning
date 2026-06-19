# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Create sample car dataset
data = {
    'Year': [2015, 2016, 2017, 2018, 2019, 2020, 2021],
    'Kilometers': [80000, 70000, 60000, 50000, 40000, 30000, 20000],
    'Engine': [1200, 1300, 1400, 1500, 1600, 1800, 2000],
    'Price': [300000, 350000, 450000, 550000, 650000, 800000, 950000]
}

# Convert data into DataFrame
df = pd.DataFrame(data)

# Input features and target
X = df[['Year', 'Kilometers', 'Engine']]
y = df['Price']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

# Create Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("Car Price Prediction")
print("--------------------")
print("Mean Absolute Error:",
      mean_absolute_error(y_test, y_pred))

print("R2 Score:",
      r2_score(y_test, y_pred))

# Predict new car price
new_car = [[2022, 15000, 2000]]

price = model.predict(new_car)

print("\nCar Details:")
print("Year = 2022")
print("Kilometers = 15000")
print("Engine = 2000 cc")

print("Predicted Price: ₹", int(price[0]))
