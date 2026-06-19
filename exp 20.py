# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


# Create Sales Dataset
data = {
    'Month': [1, 2, 3, 4, 5, 6, 7, 8],
    'Previous_Sales': [200, 250, 300, 350, 400, 450, 500, 550],
    'Advertisement': [10000, 12000, 15000, 18000, 20000, 25000, 30000, 35000],
    'Product_Price': [100, 95, 90, 85, 80, 75, 70, 65],
    'Future_Sales': [220, 280, 330, 390, 450, 520, 580, 650]
}


# Convert dataset into DataFrame
df = pd.DataFrame(data)


# Display dataset
print("Sales Dataset:")
print(df)


# Features and target
X = df[['Month','Previous_Sales',
        'Advertisement','Product_Price']]

y = df['Future_Sales']


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=1
)


# Create Linear Regression model
model = LinearRegression()


# Train model
model.fit(X_train, y_train)


# Prediction
y_pred = model.predict(X_test)


# Evaluate model
print("\nFuture Sales Prediction")
print("----------------------")

print("Mean Absolute Error:",
      mean_absolute_error(y_test,y_pred))

print("R2 Score:",
      r2_score(y_test,y_pred))


# Predict future sales
new_sales = pd.DataFrame(
    [[9,600,40000,60]],
    columns=[
        'Month',
        'Previous_Sales',
        'Advertisement',
        'Product_Price'
    ]
)


prediction = model.predict(new_sales)


print("\nNew Sales Details:")
print(new_sales)

print("Predicted Future Sales:",
      int(prediction[0]), "units")
