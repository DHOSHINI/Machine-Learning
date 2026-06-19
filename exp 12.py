# Import required libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load Iris dataset
iris = load_iris()

# Features and target
X = iris.data
y = iris.target

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

# Create KNN classifier
k = 3
knn = KNeighborsClassifier(n_neighbors=k)

# Train the model
knn.fit(X_train, y_train)

# Prediction
y_pred = knn.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("KNN Iris Flower Classification")
print("--------------------------------")
print("K Value:", k)
print("Accuracy:", accuracy * 100, "%")

# Classification report
print("\nClassification Report:")
print(classification_report(
    y_test, 
    y_pred, 
    target_names=iris.target_names
))

# Predict new flower
sample = [[5.1, 3.5, 1.4, 0.2]]

prediction = knn.predict(sample)

print("New Flower Details:")
print(sample)

print("Predicted Class:",
      iris.target_names[prediction][0])
