import numpy as np

print('Enter dimension n: ')
n = int(input())

A = np.random.randint(low=-1000, high=1000, size=(n, n))
while np.linalg.det(A) == 0:
    A = np.random.randint(low=-1000, high=1000, size=(n, n))

eigen_values, eigen_vectors = np.linalg.eig(A)
print(f'\nMatrix A:\n{A}\n')
print(f'Eigen values:\n{eigen_values}\n')
print(f'Eigen vectors:\n{eigen_vectors}\n')

"reconstruct the matrix from eigenvalues and eigenvectors"
A_reconstructed = np.dot(eigen_vectors, np.dot(np.diag(eigen_values), np.linalg.inv(eigen_vectors)))

print(f'\nReconstructed matrix:\n{A_reconstructed}\n')

rtol = 1e-05
atol = 1e-08

ans = np.allclose(A, A_reconstructed, rtol, atol)
print(f'Are the matrices equal? - {ans}')