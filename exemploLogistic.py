import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from classifier.LogisticClassifier import *
import matplotlib.pyplot as plt

print("DIABETES")
data = pd.read_csv('datasets/4-diabetes/pima-indians-diabetes.data', header=None)
y = data[8].values     
X = preprocessing.scale(data.drop([8], axis = 1).values)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

print("REGRESSÃO LOGÍSTICA")
clf = LogisticClassifier(lr = 0.1)
clf.train(X_train, y_train, epochs = 3)
print("ACURÁCIA {:.2f}%".format(clf.score(X_test, y_test)*100.0))
print("MATRIZ DE CONFUSÃO")
print(clf.confusion_matrix(X_test, y_test))

print("TESTANDO A INFLUÊNCIA DO NÚMERO DE ÉPOCAS")
ind = np.arange(1, 100, 2)
acc = []
for i in ind:
    clf = LogisticClassifier(lr = 0.1)
    clf.train(X_train, y_train, epochs = i)
    acc.append(clf.score(X_test, y_test)*100.0)
plt.plot(ind, acc, 'o-')
plt.title("Influência do número de épocas com a Acurácia")
plt.xlabel("Número de épocas")
plt.ylabel("Acurácia (%)")
plt.show()
