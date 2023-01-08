import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
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

model = MultinomialNB()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

N = 10

selector = SelectKBest(f_classif, k=N)

# Fit the SelectKBest object to the data and get the top N features
X_new = selector.fit_transform(X_train, y_train)

# Get the indices of the top N features
top_N_features = selector.get_support(indices=True)

X_train = X_train.iloc[:, top_N_features]
X_test = X_test.iloc[:, top_N_features]

model = MultinomialNB()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy on test dataset:", accuracy)

predictions = model.predict(X_train)
accuracy = accuracy_score(y_train, predictions)
print("Accuracy on train dataset:", accuracy)

