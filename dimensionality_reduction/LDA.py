import numpy as np
from numpy import linalg as LA

class LDA():
    def __init__(self, dim = 2):
        self.dim = dim
        self.matrixTransf = None
        
    def fit_transform(self, X, labels):
    
        positive = []
        negative = []
        for i in range(len(labels)):
            if labels[i] == 1:
                positive.append(X[i])
            else:
                negative.append(X[i])
        
        positive = np.array(positive)
        negative = np.array(negative)
        
        media_pos = np.mean(positive, axis = 0)
        media_neg = np.mean(negative, axis = 0)
        cov_pos = np.cov(positive.T)
        cov_neg = np.cov(negative.T)
        
        SW = cov_pos + cov_neg
        sub = (media_pos - media_neg)
        
        print(SW.shape)
        print(sub.shape)
        
        wLDA = np.matmul(LA.pinv(SW), sub)
        
        self.matrixTransf = np.array(wLDA)
        print("Matriz de transformação")
        print(self.matrixTransf)
        
        res = np.matmul(X, self.matrixTransf.T)
        return res
