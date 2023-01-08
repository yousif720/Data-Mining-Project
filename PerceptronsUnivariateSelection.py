from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif

table = pd.read_csv('test.csv', sep=',', header=None).values
attribute_names = table[0,0:14]
X_test = pd.read_csv('test.csv', sep=',', header=None).iloc[1:,:14]

element_array = table[1:,-1].ravel()
y_test = []

for x in element_array:
    if (x == '>50K'):
        y_test.append(1)
    else:
        y_test.append(0)

y_test = np.array(y_test)

table = pd.read_csv('train.csv', sep=',', header=None).values
attribute_names = table[0,0:14]
X_train = pd.read_csv('train.csv', sep=',', header=None).iloc[1:,:14]

element_array = table[1:,-1].ravel()
y_train = []

for x in element_array:
    if (x == '>50K'):
        y_train.append(1)
    else:
        y_train.append(0)

y_train = np.array(y_train)


for col in X_test.columns:
    X_test[col] = LabelEncoder().fit_transform(X_test[col])

for col in X_train.columns:
    X_train[col] = LabelEncoder().fit_transform(X_train[col])

N = 10

selector = SelectKBest(f_classif, k=N)

# Fit the SelectKBest object to the data and get the top N features
X_new = selector.fit_transform(X_train, y_train)

# Get the indices of the top N features
top_N_features = selector.get_support(indices=True)

X_train = X_train.iloc[:, top_N_features]
X_test = X_test.iloc[:, top_N_features]

# Create an MLP classifier
mlp = MLPClassifier(hidden_layer_sizes=(5, 5, 5, 5), max_iter=1000)

# Train the classifier
mlp.fit(X_train, y_train)
# Test the classifier
accuracy = mlp.score(X_test, y_test)
print("Accuracy on test: {:.2f}".format(accuracy))

accuracy = mlp.score(X_train, y_train)
print("Accuracy on train: {:.2f}".format(accuracy))