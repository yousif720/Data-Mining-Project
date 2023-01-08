import scipy.io
import numpy as np
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from toolbox.plot_boundaries import plot_boundaries


class_names = ['<=50K','>50k']

table = pd.read_csv('census.csv', sep=',', header=None).values
attribute_names = table[0,0:14]
X = pd.read_csv('census.csv', sep=',', header=None).iloc[1:,:14]
X = pd.DataFrame(X.values,columns = attribute_names)
X = X.drop(['fnlwgt','native-country','workclass','race'], axis = 1)

element_array = table[1:,-1].ravel()
y = []

for x in element_array:
    if (x == '>50K'):
        y.append(1)
    else:
        y.append(0)

y = np.array(y)

for col in X.columns:
    X[col] = LabelEncoder().fit_transform(X[col])

kf = KFold(n_splits=10)
avg_train = []
avg_test = []

for i in range(19):
    avg_train.append([])
    avg_test.append([])

for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]

    for depth in range(19):
        dtc = tree.DecisionTreeClassifier(max_depth = depth + 2)
        dtc = dtc.fit(X_train, y_train)
        avg_train[depth].append(1 - dtc.score(X_train, y_train))
        avg_test[depth].append(1 - dtc.score(X_test, y_test))

for i in range(19):
    avg_train[i] = np.mean(avg_train[i])
    avg_test[i] = np.mean(avg_test[i])
    
print("Average training error: ", avg_train)
print("Average testing error: ", avg_test)

print("Average training error: ", avg_train)
print("Average testing error: ", avg_test)

print("Lowest training error: ", np.min(avg_train), "is at index: ", np.argmin(avg_train))
print("Lowest testing error: ", np.min(avg_test), "is at index: ", np.argmin(avg_test))

#plt.plot(range(2,21), avg_train, label = "Training errors")
#plt.plot(range(2,21), avg_test, label = "Testing errors")
#plt.legend()
#plt.show()

