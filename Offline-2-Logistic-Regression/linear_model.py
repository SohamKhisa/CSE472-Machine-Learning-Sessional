import numpy as np

class LogisticRegression:
    def __init__(self, params):
        """
        figure out necessary params to take as input
        :param params:
        """
        # todo: implement
        self.alpha = params['learning_rate']
        self.iter = params['max_iter']

    def fit(self, X, y):
        """
        :param X:
        :param y:
        :return: self
        """
        assert X.shape[0] == y.shape[0]
        assert len(X.shape) == 2
        # todo: implement
        self.X = X.to_numpy().reshape(X.shape[0], X.shape[1])
        self.y = y.to_numpy().reshape(y.shape[0], 1)
        m, n = X.shape
        self.W = np.zeros((n,1))
        self.b = 0

        for i in range(self.iter):
            z = np.dot(self.X, self.W) + self.b         # z = X * W + b; ***X->(m,n), W->(n,1)***
            A = 1 / (1 + np.exp(-z))                    # A = sigmoid(z); ***A->(m,1)***
            dz = A - self.y                             # dz = A - y; ***dz->(m,1)***
            dW = 1/m * np.dot(self.X.T, dz)             # dW = X.T * dz; ***X->(m,n), dz->(m,1), dW->(n,1)***
            db = 1/m * np.sum(dz)                       # db = sum(dz); ***dz->(m,1), db->(1,1)***
            self.W = self.W - self.alpha * dW           # W = W - alpha * dW; ***W->(n,1), dW->(n,1), dW->(n,1), alpha->learning_rate***
            self.b = self.b - self.alpha * db           # b = b - alpha * db; ***b->(1,1), db->(1,1), db->(1,1), alpha->learning_rate***

        return self


    def predict(self, X):
        """
        function for predicting labels of for all datapoint in X
        :param X:
        :return:
        """
        # todo: implement
        z = np.dot(X, self.W) + self.b
        A = 1 / (1 + np.exp(-z))
        y_pred = np.where(A > 0.5, 1, 0)
        return y_pred