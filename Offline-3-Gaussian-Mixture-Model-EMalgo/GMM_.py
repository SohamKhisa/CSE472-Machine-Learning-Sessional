import numpy as np
from scipy.stats import multivariate_normal

class GMM:
    def __init__(self, k, max_iter=100, tol=1e-3):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        
    def fit(self, X):
        n, d = X.shape
        self.means = np.random.rand(self.k, d)
        self.covs = np.array([np.eye(d) for _ in range(self.k)])
        self.weights = np.ones(self.k) / self.k
        self.responsibilities = np.zeros((n, self.k))
        
        for i in range(self.max_iter):
            prev_log_likelihood = self.log_likelihood(X)
            
            # E-Step
            self.responsibilities = self._compute_responsibilities(X)
            
            # M-Step
            self._update_params(X)
        
            #print parameters
            

            log_likelihood = self.log_likelihood(X)
            if abs(log_likelihood - prev_log_likelihood) < self.tol:
                break
                
    def _compute_responsibilities(self, X):
        n, d = X.shape
        responsibilities = np.zeros((n, self.k))
        for i in range(self.k):
            responsibilities[:, i] = self.weights[i] * multivariate_normal.pdf(X, mean=self.means[i], cov=self.covs[i], allow_singular=True)
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)
        return responsibilities
    
    def _update_params(self, X):
        n, d = X.shape
        for i in range(self.k):
            resp = self.responsibilities[:, i]
            
            

            if resp.sum() < 1e-10:
                self.means[i] = np.random.rand(d)
                self.weights[i] = 1 / self.k
                self.covs[i] = np.eye(d)
                self.responsibilities[:, i] = 1 / self.k
            else:
                self.covs[i] = np.cov(X.T, aweights=resp, bias=True)
                self.means[i] = (resp[:, np.newaxis] * X).sum(axis=0) / resp.sum()
                self.weights[i] = resp.sum() / n

            

            
    def log_likelihood(self, X):
        n, d = X.shape
        log_likelihood = 0
        for i in range(self.k):
            log_likelihood += self.weights[i] * multivariate_normal.pdf(X, mean=self.means[i], cov=self.covs[i], allow_singular=True)
        return np.log(log_likelihood).sum()
