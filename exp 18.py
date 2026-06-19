# Import required libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, classification_report


# Load Iris dataset
iris = load_iris()

# Features and target
X = iris.data
y = iris.target


# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=1
)


# Create Perceptron model
model = Perceptron(
    max_iter=1000,
    eta0=0.1,
    random_state=1
)


# Train the model
model.fit(X_train, y_train)


# Predict test data
y_pred = model.predict(X_test)


# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)


print("Perceptron Based Iris Classification")
print("-----------------------------------")

print("Accuracy:",
      accuracy * 100, "%")


# Classification report
print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=iris.target_names
))


# Predict new flower
new_flower = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],
    columns=[
        'Sepal Length',
        'Sepal Width',
        'Petal Length',
        'Petal Width'
    ]
)

prediction = model.predict(new_flower)


print("New Flower Details:")
print(new_flower)

print("Predicted Class:",
      iris.target_names[prediction][0])
