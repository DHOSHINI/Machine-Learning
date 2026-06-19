import pandas as pd
import numpy as np

# Read CSV file
data = pd.read_csv("data.csv")

concepts = np.array(data.iloc[:, :-1])
target = np.array(data.iloc[:, -1])

def candidate_elimination(concepts, target):

    specific_h = concepts[0].copy()

    general_h = [["?" for i in range(len(specific_h))]
                 for i in range(len(specific_h))]

    print("\nInitial Specific Hypothesis:")
    print(specific_h)

    print("\nInitial General Hypothesis:")
    print(general_h)

    for i, h in enumerate(concepts):

        if target[i] == "Yes":

            for x in range(len(specific_h)):

                if h[x] != specific_h[x]:
                    specific_h[x] = '?'
                    general_h[x][x] = '?'

        elif target[i] == "No":

            for x in range(len(specific_h)):

                if h[x] != specific_h[x]:
                    general_h[x][x] = specific_h[x]
                else:
                    general_h[x][x] = '?'

        print("\nStep", i + 1)
        print("Specific Boundary:", specific_h)
        print("General Boundary:", general_h)

    general_h = [g for g in general_h if g != ['?'] * len(specific_h)]

    return specific_h, general_h

s_final, g_final = candidate_elimination(concepts, target)

print("\nFinal Specific Hypothesis:")
print(s_final)

print("\nFinal General Hypothesis:")
for g in g_final:
    print(g)
