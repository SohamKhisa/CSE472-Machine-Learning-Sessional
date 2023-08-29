import numpy as np
from data_handler import bagging_sampler
from linear_model import LogisticRegression

class BaggingClassifier:
    def __init__(self, base_estimator, n_estimator):
        """
        :param base_estimator:
        :param n_estimator:
        :return:
        """
        # todo: implement
        self.estimators = []
        self.estimator = base_estimator
        self.n_estimator = n_estimator

    def fit(self, X, y):
        """
        :param X:
        :param y:
        :return: self
        """
        assert X.shape[0] == y.shape[0]
        assert len(X.shape) == 2
        # todo: implement
        # the first model was not trained previously
        for i in range(self.n_estimator):
            X_train, y_train = bagging_sampler(X, y)
            self.estimators.append(self.estimator.fit(X_train, y_train))

    def predict(self, X):
        """
        function for predicting labels of for all datapoint in X
        apply majority voting
        :param X:
        :return:
        """
        # todo: implement
        y_pred = np.zeros((X.shape[0], 1))
        for estimator in self.estimators:
            y_pred += estimator.predict(X)
        y_pred = np.where(y_pred > self.n_estimator/2, 1, 0)
        return y_pred
