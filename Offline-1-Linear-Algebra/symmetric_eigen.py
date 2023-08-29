import numpy as np

print('Enter dimension n: ')
n = int(input())

A = np.zeros((n, n))
while np.linalg.det(A) == 0:
    A = np.random.randint(low=-1000, high=1000, size=(n, n))
    A = A + A.T

eigen_values, eigen_vectors = np.linalg.eig(A)
print(f'\nMatrix A:\n{A}\n')
print(f'Eigen values:\n{eigen_values}\n')
print(f'Eigen vectors:\n{eigen_vectors}\n')

"reconstruct the matrix from eigenvalues and eigenvectors"
# Because for a symmetric matrix, the eigenvalues are always real
# and the corresponding eigenvectors are always orthogonal, thus mat.T = mat.inv 

A_reconstructed = np.dot(eigen_vectors, np.dot(np.diag(eigen_values), eigen_vectors.T))

print(f'\nReconstructed matrix:\n{A_reconstructed}\n')

rtol = 1e-05
atol = 1e-09

ans = np.allclose(A, A_reconstructed, rtol, atol)
print(f'Are the matrices equal? - {ans}')