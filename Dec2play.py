import scipy.io
import numpy as np
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold


class_names = ['<=50K','>50k']

table = pd.read_csv('census.csv', sep=',', header=None).values
attribute_names = table[0,0:14]
X = pd.read_csv('census.csv', sep=',', header=None).iloc[1:,:14]

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

dtc = tree.DecisionTreeClassifier(max_depth = 9)
dtc = dtc.fit(X, y)

# YOUR CODE HERE
accuracy = dtc.score(X, y)
print(accuracy)
plt.figure(figsize=[50,50])
tree.plot_tree(dtc, feature_names = attribute_names, class_names = class_names)
plt.show()