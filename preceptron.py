import numpy as np

class Preceptron():
    def __init__(self, lr=0.1, max_iter=1000):
        self.lr = lr
        self.max_iter = max_iter
        self.weights = np.random.uniform(-0.5, 0.5, 3) # 2 weights and a bias
    
    def activation(self, inputs):
        # Use dot product as activation function 
        y = np.dot(self.weights, inputs)
        # Based on result return 0 or 1
        result = 0
        if y > 0:
            result = 1
        return result
    
    def predict(self, X):
        X = np.insert(X, 0, 1) # add bias
        return self.activation(X)

    def train(self, X, y):
        for _ in range(self.max_iter): # Do this only max 1000 times
            err_flag = False
            for i in range(len(X)): # Loop through data (both are same length)
                result = self.predict(X[i]) # Predict results to use for error findings
                error = y[i] - result # Calculate error
                if error != 0:
                    self.weights += self.lr * error * np.insert(X[i], 0, 1) # update weights
                    err_flag = True
            if not err_flag: # Stop updating when no errors are found
                break