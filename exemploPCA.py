import pandas as pd
from sklearn.model_selection import train_test_split
from dimensionality_reduction.PCA import *
import matplotlib.pyplot as plt


print("DIABETES")
data = pd.read_csv('datasets/4-diabetes/pima-indians-diabetes.data', header=None)
labels = data[8].values     
X = data.drop([8], axis = 1).values

clf = PCA(2)
y = clf.fit_transform(X)
for _x, _y in zip(X, y):
    print("{} -> {}".format(_x, _y))

plt.scatter(y[:, 0], y[:, 1], c=labels)
plt.title("Dados transformados")
plt.show()
