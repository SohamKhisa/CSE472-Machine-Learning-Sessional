import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import multivariate_normal

class GMM:
    def __init__(self, params):
        self.data = params['data']
        self.max_iter = params['max_iter'] if 'max_iter' in params else 100
    
    def regularize_covariance(self, sigma, eps=1e-6):
        for k in range(sigma.shape[0]):
            sigma[k] += eps * np.eye(sigma[k].shape[0])
    
    def animate(self, k, iter):
        plt.cla()
        plt.xlabel('X')
        plt.ylabel("Y")
        plt.scatter(self.data[:, 0], self.data[:, 1], s=5)
        x = np.linspace(self.data[:, 0].min()*1.3, self.data[:, 0].max()*1.3, 100)
        y = np.linspace(self.data[:, 1].min()*1.3, self.data[:, 1].max()*1.3, 100)
        X, Y = np.meshgrid(x, y)
        for i in range(k):
            pos = np.empty(X.shape + (2,))
            pos[:, :, 0] = X
            pos[:, :, 1] = Y
            Z = multivariate_normal.pdf(pos, self.mu[i], self.sigma[i])
            plt.contour(X, Y, Z, levels=10, cmap='viridis')
            plt.title('Task-2/3: Iteration ' + str(iter) + ', k = ' + str(k))
        plt.pause(0.1)
    
    def fit(self, k, iskstar=False):
        # implementation of em algorithm
        n, m = self.data.shape

        # initialize parameters
        # mu -> (kxm)
        # sigma -> k number of mxm matrices
        self.phi = np.ones(k) / k
        self.mu = np.random.rand(k, m)
        self.sigma = np.array([np.eye(m) for _ in range(k)])
        loglikelihood = None
        
        for iter in range(self.max_iter):
            #********E-step**** *****#
            # z -> latent variable (nxk)

            z = np.zeros((n, k))
            for i in range(k):
                z[:, i] = self.phi[i] * multivariate_normal.pdf(self.data, mean=self.mu[i], cov=self.sigma[i], allow_singular=True)
            z /= np.sum(z, axis=1, keepdims=True)

            #********M-step*********#
            # update parameters

            zk = np.sum(z, axis=0, keepdims=True)
                
            for i in range(k):
                # the shape of z[:, i] is (n,) which is a rank 1 array.
                # zk->(1xk)
                self.mu[i] = np.sum(z[:, i].reshape(-1, 1) * self.data, axis=0) / zk[0][i]
                centered_data = self.data - self.mu[i]
                self.sigma[i] = np.dot(centered_data.T, centered_data * z[:, i].reshape(-1, 1)) / zk[0][i]
                self.phi[i] = np.sum(z[:, i])/n
            
            new_loglikelihood = 0
            for i in range(k):
                new_loglikelihood += self.phi[i] * multivariate_normal.pdf(self.data, mean=self.mu[i], cov=self.sigma[i], allow_singular=True)
            new_loglikelihood = np.log(new_loglikelihood).sum()
            self.regularize_covariance(self.sigma)

            # task-2
            if loglikelihood is not None and iskstar and m<=2:
                self.animate(k, iter)

            if loglikelihood is not None and np.abs(new_loglikelihood - loglikelihood) < 1e-6:
                break
            loglikelihood = new_loglikelihood
        print("k: {}, loglikelihood: {}\n".format(k, loglikelihood))
        return loglikelihood