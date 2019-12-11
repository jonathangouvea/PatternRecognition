import pandas as pd
from sklearn.model_selection import train_test_split
from classifier.NaiveBayesClassifier import *
import matplotlib.pyplot as plt

print("WINE")

data = pd.read_csv('datasets/8-wine/wine.data', header=None)
#print(data.head())
y = data[0].values
X = data.drop([0], axis = 1).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

print("NAIVE BAYES DO CASO 1")
clf = NaiveBayesClassifier(caso = 1)
clf.train(X_train, y_train)
print("ACURÁCIA {:.2f}%".format(clf.score(X_test, y_test)*100.0))
print("MATRIZ DE CONFUSÃO")
print(clf.confusion_matrix(X_test, y_test))

print("\nNAIVE BAYES DO CASO 2")
clf = NaiveBayesClassifier(caso = 2)
clf.train(X_train, y_train)
print("ACURÁCIA {:.2f}%".format(clf.score(X_test, y_test)*100.0))
print("MATRIZ DE CONFUSÃO")
print(clf.confusion_matrix(X_test, y_test))

print("\nNAIVE BAYES DO CASO 3")
clf = NaiveBayesClassifier(caso = 3)
clf.train(X_train, y_train)
print("ACURÁCIA {:.2f}%".format(clf.score(X_test, y_test)*100.0))
print("MATRIZ DE CONFUSÃO")
print(clf.confusion_matrix(X_test, y_test))




print("\n\nDERMATOLOGY")

data = pd.read_csv('datasets/3-dermatology/dermatology.data', header=None)
#print(data.head())
y = data[34].values
X = data.drop([34], axis = 1).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

clf = NaiveBayesClassifier(caso = 3)
clf.train(X_train, y_train)
print("ACURÁCIA {:.2f}%".format(clf.score(X_test, y_test)*100.0))
print("MATRIZ DE CONFUSÃO")
print(clf.confusion_matrix(X_test, y_test))
