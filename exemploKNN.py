import pandas as pd
from sklearn.model_selection import train_test_split
from classifier.KNNClassifier import *
import matplotlib.pyplot as plt

print("WINE")

data = pd.read_csv('datasets/8-wine/wine.data', header=None)
#print(data.head())
y = data[0].values
X = data.drop([0], axis = 1).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

knn = KNNClassifier(3)
knn.train(X_train, y_train)
print("ACURÁCIA {:.2f}%".format(knn.score(X_test, y_test)*100.0))
print("MATRIZ DE CONFUSÃO")
print(knn.confusion_matrix(X_test, y_test))



print("\n\nDERMATOLOGY")

data = pd.read_csv('datasets/3-dermatology/dermatology.data', header=None)
#print(data.head())
y = data[34].values
X = data.drop([34], axis = 1).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

knn = KNNClassifier(3)
knn.train(X_train, y_train)
print("ACURÁCIA {:.2f}%".format(knn.score(X_test, y_test)*100.0))
print("MATRIZ DE CONFUSÃO")
print(knn.confusion_matrix(X_test, y_test))


print("\n\nTESTANDO DIFERENTES KNN")

K = np.arange(3, 50, 2)
acc = []
for k in K:
    knn = KNNClassifier(k)
    knn.train(X_train, y_train)
    acc.append(knn.score(X_test, y_test) * 100.0)
plt.plot(K, acc, marker = 'o')
plt.ylim((0, 100))
plt.title("Acurácia relacionada com mudança do K")
plt.xlabel("K")
plt.ylabel("Acurácia (%)")
plt.show()
