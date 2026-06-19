import pandas as pd

# Read dataset
data = pd.read_csv("training_data.csv")

print("Training Dataset:")
print(data)


# Separate attributes and target
X = data.iloc[:, :-1]
Y = data.iloc[:, -1]


# Initialize hypothesis
hypothesis = ['0'] * len(X.columns)

print("\nInitial Hypothesis:")
print(hypothesis)


# FIND-S Algorithm
for i in range(len(X)):

    # Consider only positive examples
    if Y[i] == "Yes":

        for j in range(len(X.columns)):

            # First positive example
            if hypothesis[j] == '0':
                hypothesis[j] = X.iloc[i, j]

            # Generalize hypothesis
            elif hypothesis[j] != X.iloc[i, j]:
                hypothesis[j] = '?'

        print("\nAfter example", i+1)
        print(hypothesis)


print("\nFinal Most Specific Hypothesis:")
print(hypothesis)
