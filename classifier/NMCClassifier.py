import numpy as np

class NMCClassifier():
    def __init__(self):
        self.X = None
        self.y = None
        self.uniqueY = None
        self.hasZero = False

    def train(self, X, y):

        if len(X) != len(y):
            print("ERRO: Tamanho de X e Y são diferentes")
            print(len(X))
            print(len(y))
            return
            
        self.X = X
        self.y = y
        self.uniqueY = np.unique(y)
        if 0 in self.uniqueY:
            self.hasZero = True

    def predict(self, X):
        y = []
        for x in X:
            predict_y = np.zeros(len(self.uniqueY))
            dist = np.zeros(len(self.X))
            i = 0
            for sx in self.X:
                dist[i] = np.linalg.norm(x - sx)
                i += 1
            dist_sort = np.argsort(dist)
            
            if self.hasZero:
                predict_y[ self.y[ dist_sort[0] ]] += 1
                y.append(np.argmax(predict_y))
            else:
                predict_y[ self.y[ dist_sort[0] ] - 1] += 1
                y.append(np.argmax(predict_y) + 1)
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
        cm = np.zeros((len(self.uniqueY), len(self.uniqueY)), int)
        for i in range(len(predict_y)):
            cm[predict_y[i] - 1, true_y[i] - 1] += 1
        return cm

