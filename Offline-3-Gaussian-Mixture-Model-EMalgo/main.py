import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data_handler import load_dataset
from gmm import GMM

data = load_dataset()
params = {'data': data, "max_iter": 100}
log_likelihoods = []
k_range = range(1, 11)
for k in k_range:
    gmm = GMM(params)
    log_likelihoods.append(gmm.fit(k))

plt.figure()
sns.lineplot(x=k_range, y=log_likelihoods)
plt.title("task-1")
plt.xlabel('Number of components')
plt.ylabel('Converged log-likelihood')
plt.show()

# diff = np.diff(log_likelihoods)
# k_star = np.argmin(diff) + 1
print("\n Enter the value of K*: ")
k_star = input()
k_star = int(k_star)
print(f'k* = {k_star}')
gmm = GMM(params)
plt.ion()
gmm.fit(k_star, iskstar=True)
plt.ioff()
plt.show()