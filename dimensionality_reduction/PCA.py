import numpy as np
from numpy import linalg as LA

class PCA():
    def __init__(self, dim = 2):
        self.dim = dim
        self.matrixTransf = None
        
    def fit_transform(self, X):
        media = np.mean(X, axis = 1)
        cov = np.cov(X.T)
        
        v, w = LA.eig(cov)
        print("AUTOVALORES")
        print(v)
        print("AUTOVETORES")
        print(w)
        print()
        
        ordem = np.argsort(v)
        
        wPCA = []
        for i in range(self.dim):
            ind = ordem[len(ordem) - i - 1]
            
            wPCA.append(w[ind])
            
        self.matrixTransf = np.array(wPCA)
        print("Matriz de transformação")
        print(self.matrixTransf)
        
        res = np.matmul(X, self.matrixTransf.T)
        return res
