import numpy as np

class KMeans():
    def __init__(self, k = 3):
        self.k = k
        self.centers = None
        
    def fit(self, X):
        centers = np.random.normal(0, 1, (self.k, X.shape[1]))
        
        
        dists = []
        for x in X:
            dist = []
            for c in centers:
                dist.append(np.linalg.norm(x - c))
            dists.append(dist)
        
        y = np.argmin(dists, axis = 1)
        print("Estado inicial\nCentróides aleatórios")
        print(centers)
        
        mudou = True
        repet = 0
        while mudou and repet < 50:
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
            
            '''print("Iteração {}\nCentróides aleatórios".format(repet))
            print(centers)
            print("Predição")
            print(y2)
            print("\n")'''
            
            y = y2
            
        print("Iteração {}\nCentróides aleatórios".format(repet))
        print(centers)
        print("\n")
        self.centers = centers
        return centers
        
    def predict(self, X):
        y = []
        for x in X:
            dist = []
            for c in self.centers:
                dist.append(np.linalg.norm(x - c))
            y.append(np.argmin(dist))
        return y
        
        
        
