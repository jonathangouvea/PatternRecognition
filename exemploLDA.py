import pandas as pd
from sklearn.model_selection import train_test_split
from dimensionality_reduction.LDA import *
from dimensionality_reduction.PCA import *
import matplotlib.pyplot as plt


print("DIABETES")
data = pd.read_csv('datasets/4-diabetes/pima-indians-diabetes.data', header=None)
labels = data[8].values     
X = data.drop([8], axis = 1).values

clf = LDA(1)
y = clf.fit_transform(X, labels)
for _x, _y in zip(X, y):
    print("{} -> {}".format(_x, _y))

print("COMPARAÇÃO")

fig, ax = plt.subplots(1, 2, figsize=(10, 10))

pca = PCA(1)
y = pca.fit_transform(X)
ax[0].set_aspect('equal')
ax[0].set_title('PCA')
ax[0].scatter(y[:, 0], y[:, 0], c=labels)

lda = LDA(1)
y = lda.fit_transform(X, labels)
ax[1].set_aspect('equal')
ax[1].set_title('LDA')
ax[1].scatter(y, y, c=labels)

fig.suptitle("Dados transformados")
plt.tight_layout()
plt.show()
