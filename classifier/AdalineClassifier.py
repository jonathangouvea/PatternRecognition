import numpy as np

class AdalineClassifier():
    def __init__(self, lr = 0.001):
        self.weights = None
        self.alpha = 0
        self.lr = lr
        
    def train(self, X, y, epochs = 3):
        self.weights = np.zeros(X.shape[1])
        self.alpha = 0
        for _ in range(epochs):
            pred = np.inner(X, self.weights) + self.alpha
            
            erros = (y - pred)
            self.alpha += self.lr * erros.sum()
            self.weights += self.lr * np.dot(X.T, erros)
            
    def predict(self, X):
        y = []
        for x in X:
            z = np.inner(self.weights, x) + self.alpha
            
            if z <= 0:
                y.append(0)
            else:
                y.append(1)
        return y
        
    def score(self, X, true_y):
        predict_y = self.predict(X)
        total = len(predict_y)
        certos = 0
        for i in range(total):
            if predict_y[i] == true_y[i]:
                certos += 1
        return certos/total
        
    def confusion_matrix(self, X, true_y):
        predict_y = self.predict(X)
        cm = np.zeros((2, 2), int)
        for i in range(len(predict_y)):
            cm[predict_y[i], true_y[i]] += 1
        return cm
