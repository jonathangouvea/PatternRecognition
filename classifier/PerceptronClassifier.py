import numpy as np

class PerceptronClassifier():
    def __init__(self, lr = 0.1):
        self.weights = None
        self.alpha = 0
        self.lr = lr
        
    def train(self, X, y, epochs = 3):
        self.weights = np.zeros(X.shape[1])
        self.alpha = 0
        for _ in range(epochs):
            for i in range(len(y)):
                z = np.inner(self.weights, X[i]) - self.alpha
                g = 1
                if z < 0:
                    g = -1
                self.alpha += self.lr * (y[i] - g)
                for j in range(len(self.weights)):
                    self.weights[j] += self.lr * (y[i] - g) * X[i][j]
        
    def predict(self, X):
        y = []
        for x in X:
            z = np.inner(self.weights, x) - self.alpha
            if z < 0:
                y.append(-1)
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
            if predict_y[i] == 1:
                if true_y[i] == 1:
                    cm[1, 1] += 1
                else:
                    cm[1, 0] += 1
            else:
                if true_y[i] == 1:
                    cm[0, 1] += 1
                else:
                    cm[0, 0] += 1
        return cm
