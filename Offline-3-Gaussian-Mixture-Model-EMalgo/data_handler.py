import numpy as np
import pandas as pd
import sys

from sklearn.decomposition import PCA

def load_dataset():
    if len(sys.argv) < 2:
        print('No file name provided')
        sys.exit(1)
    file = sys.argv[1]

    data = pd.read_csv(file, sep=" ")
    data = data.values

    _, m = data.shape
    if m > 2:
        pca = PCA(n_components=2)
        data2D = pca.fit_transform(data)
        data = pd.DataFrame(data2D)
        data = data.values
    return data