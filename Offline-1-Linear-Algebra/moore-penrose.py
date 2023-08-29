import numpy as np

print('Enter dimensions')
print('Enter n: ')
n = int(input())
print('Enter m: ')
m = int(input())

A = np.random.randint(low=-1000, high=1000, size=(n, m))
u, d, vt = np.linalg.svd(A, full_matrices=True)

print(f'\nMatrix A:\n{A}\n')
print(f'U:\n{u}\n\nD:\n{d}\n\nV:\n{vt}\n')

d_plus = np.diag(1/d)
if (m >= n):
    d_plus = np.append(d_plus, np.zeros((m-n, n)), axis=0)
else:
    d_plus = np.append(d_plus, np.zeros((m, n-m)), axis=1)

print(f'shape of A: {A.shape}')
print(f'shape of d_plus: {d_plus.shape}\n')

# vt.T = v
# D+ = 1/D, (A->nxm), (D+ -> mxn)
pinv_nump = np.linalg.pinv(A)
pinv_eq = np.dot(vt.T, np.dot(d_plus, u.T))

print(f'\nMoore-Penrose pseudo-inverse using numpy:\n{pinv_nump}\n')
print(f'\nMoore-Penrose pseudo-inverse using equation:\n{pinv_eq}\n')
ans = np.allclose(pinv_nump, pinv_eq)
print(f'Are the matrices equal? - {ans}')