import numpy as np
from scipy.stats import multivariate_normal

class GMM():
    def __init__(self, k = 3):
        self.k = k
        self.centers = None
        self.proporcoes = None
        self.medias = None
        self.variancias = None
        
    def fit(self, X):
        centers = np.random.normal(0, 1, (self.k, X.shape[1]))
        
        
        dists = []
        for x in X:
            dist = []
            for c in centers:
                dist.append(np.linalg.norm(x - c))
            dists.append(dist)
        
        y = np.argmin(dists, axis = 1)
        
        mudou = True
        repet = 0
        while mudou and repet < 5:
            for i in range(len(centers)):
                x_selec = []
                for j in range(len(X)):
                    if y[j] == i:
                        x_selec.append(X[j])
                if len(x_selec) > 0:
                    centers[i] = np.mean(np.array(x_selec), axis = 0)
                
            dists = []
            for x in X:
                dist = []
                for c in centers:
                    dist.append(np.linalg.norm(x - c))
                dists.append(dist)
            
            y2 = np.argmax(dists, axis = 1)
            
            mudou = False
            for i in range(len(y)):
                if y[i] != y2[i]:
                    mudou = True
                    break
            repet += 1
            
            y = y2
            
        self.centers = centers
        medias = centers
        variancias = []
        proporcoes = []
        for i in range(self.k):
            tot = 0
            x_selec = []
            
            for j in range(len(X)):
                if y[j] == i:
                    x_selec.append(X[j])
                    tot += 1
                    
            if len(x_selec) > 0:
                variancias.append(np.cov(np.array(x_selec).T))
            else:
                variancias.append(np.zeros(X.shape[1], X.shape[1]))
            proporcoes.append(tot / len(X))
        
        print("MEDIAS\n{}\nVARIANCIAS\n{}\nPROPORCOES\n{}\n".format(medias, variancias, proporcoes))
        
        for epoc in range(50):
        
            #Passo E
            T = np.zeros((self.k, X.shape[0]))
            for j in range(self.k):
                for i in range(len(X)):
                    denominador = 0
                    for _k in range(self.k):
                        denominador += proporcoes[_k] * multivariate_normal.pdf(X[i], medias[_k], variancias[_k])
                    numerador = proporcoes[j] * multivariate_normal.pdf(X[i], medias[j], variancias[j])
                    
                    T[j, i] = numerador / denominador
            
            
            #Passo M
            for j in range(self.k):
                Nj = np.sum(T[j, :])
                uj = 1/Nj * np.dot(T[j, :], X)
                
                dim = X.shape[1]
                sigmaj = np.zeros((dim, dim))
                parcialsigma = np.zeros((dim, dim))
                for i in range(len(X)):
                    sub = np.expand_dims(X[i] - uj, axis = 1)
                    sub2 = np.dot(sub, sub.T)
                    parcialsigma = (T[j, i] * sub2)
                    
                    sigmaj += parcialsigma / Nj
                
                '''sub = X - uj
                print(np.dot(sub, sub.T))
                print((T[j, :] * np.dot(sub, sub.T)).shape)
                sigmaj = 1/Nj * np.sum(T[j, :] * np.dot(sub, sub.T), axis = 1)
                print(sigmaj.shape)
                input()'''
                
                
                propj = Nj/len(X)
                
                proporcoes[j] = propj
                medias[j] = uj
                variancias[j] = sigmaj
                
            print("ITER {}\nMEDIAS\n{}\nVARIANCIAS\n{}\nPROPORCOES\n{}\n".format(epoc, medias, variancias, proporcoes))
            
            self.proporcoes = proporcoes
            self.medias = medias
            self.variancias = variancias
            
            
        
        
    def predict(self, X):
        y = []
        for x in X:
            prob = []
            for j in range(self.k):
                prob.append(self.proporcoes[j] * multivariate_normal.pdf(x, self.medias[j], self.variancias[j]))
            y.append(np.argmax(prob))
        return y
        
    def predictKMeans(self, X):
        y = []
        for x in X:
            dist = []
            for c in self.centers:
                dist.append(np.linalg.norm(x - c))
            y.append(np.argmin(dist))
        return y
        
        
        
