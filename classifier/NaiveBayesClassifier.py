import numpy as np

class NaiveBayesClassifier():
    def __init__(self, caso = 1):
        self.mean = None
        self.variance = None
        self.probs = None
        self.ys = None
        self.caso = caso
        
    def train(self, X, y):
        y_unique = np.unique(y)
        mean = []
        variance = []
        probs = []
        ys = []
        
        for _y in y_unique:
            j = 0
            new_X = []
            for i in range(len(X)):
                if y[i] == _y:
                    new_X.append(X[i])
                    j += 1
            mean.append(np.mean(np.array(new_X).T, axis = 1))
            variance.append(np.cov(np.array(new_X).T))
            ys.append(_y)
            probs.append(j / len(y))
            
        self.mean = mean
        self.variance = variance
        self.probs = probs
        self.ys = ys
        
    def print_params(self):
        for i in range(len(self.ys)):
            print("Y {} {:.2f}%".format(self.ys[i], self.probs[i]*100.0))
            print("MÉDIA")
            print(self.mean[i])
            print("VARIANCIA")
            print(self.variance[i])
            print("\n")
        
    def predict(self, X):
        y = []
        for x in X:
            dj = []
            for i in range(len(self.ys)):                
                if self.caso == 1:
                    t1 = np.log(self.probs[i])
                
                    det = np.linalg.det(self.variance[i])
                    t2 = - 1/2 * np.log(det)
                    
                    tsub = np.subtract(x, self.mean[i])
                    t3 = np.matmul(tsub.T, np.linalg.pinv(self.variance[i]))
                    t4 = -1/2 * np.matmul(t3, tsub)
                    
                    dj.append(t1 + t2 + t4)
                    
                elif self.caso == 2:
                    det = np.linalg.det(self.variance[i])
                    t2 = - 1/2 * np.log(det)
                    
                    tsub = np.subtract(x, self.mean[i])
                    t3 = np.matmul(tsub.T, np.linalg.pinv(self.variance[i]))
                    t4 = -1/2 * np.matmul(t3, tsub)
                    
                    dj.append(t2 + t4)
                    
                else:
                    tsub = np.subtract(x, self.mean[i])
                    t3 = np.matmul(tsub.T, np.linalg.pinv(self.variance[i]))
                    t4 = -1/2 * np.matmul(t3, tsub)
                    
                    dj.append(t4)
            #print(dj)
            y.append(self.ys[np.argmax(dj)])
            
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
        cm = np.zeros((len(self.ys), len(self.ys)), int)
        for i in range(len(predict_y)):
            #print("{} -> {}x{}".format(i, predict_y[i], true_y[i]))
            cm[predict_y[i] - 1, true_y[i] - 1] += 1
        return cm
        
