import numpy as np

class Preceptron():
    def __init__(self, lr=0.1, max_iter=1000):
        self.lr = lr
        self.max_iter = max_iter
        self.weights = np.random.uniform(-0.5, 0.5, 3) # 2 weights and a bias
    
    def activation(self, inputs):
        y = np.dot(self.weights, inputs)
        result = 0
        if y > 0:
            result = 1
        return result
    
    def predict(self, X):
        X = np.insert(X, 0, 1) # add for bias
        return self.activation(X)

    def train(self, X, y):
        for _ in range(self.max_iter):
            err_flag = False
            for i in range(len(X)):
                result = self.predict(X[i])
                error = y[i] - result
                if error != 0:
                    self.weights += self.lr * error * np.insert(X[i], 0, 1)
                    err_flag = True
            if not err_flag:
                break