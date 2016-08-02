import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from Perceptron import Perceptron 
import TwoDContourMap as cm

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

X = df.iloc[0:100, [0,2]].values

#First 50 are all setosa
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
#Last 50 are all versicolor
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
plt.xlabel('sepal length')
plt.ylabel('petal length')
plt.legend(loc='upper left')
plt.show()

ppn = Perceptron(0.1, 10)
ppn.fit(X, y)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Misclassifications')
plt.show()

cm.plot_decision_regions(X,y,classifier=ppn)
plt.xlabel('sepal length')
plt.ylabel('petal length')
plt.legend(loc='upper left')
plt.show()
