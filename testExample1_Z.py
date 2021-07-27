import numpy as np

mat_A = np.loadtxt("./mat_A.txt",delimiter=",",dtype=complex)
mat_L = np.loadtxt("./mat_L.txt",delimiter=",",dtype=complex)
mat_U = np.loadtxt("./mat_U.txt",delimiter=",",dtype=complex)

print('matA')
print (mat_A)
print()
print('matL')
print (mat_L)
print()
print('matU')
print (mat_U)
print()


# solve the P.T matrix
# in scipy, A = p l u where p stands for P.T here
print('calculate P.T')
print (mat_A @ np.linalg.inv(mat_L @ mat_U))
print()

from scipy.linalg import lu

p,l,u = lu(mat_A)
print()
print('matp')
print (p)
print('matl')
print (l)
print('matu')
print (u)
print('lu')
print(l@u)
print('plu')
print(p@l@u)
print()

# reconstructing the permutation matrix from pivoting array
num_pivot = np.int32(np.loadtxt("./pivot.txt"))

x = np.eye(num_pivot.shape[0])
order0 = np.arange(num_pivot.shape[0])

y=x.copy()
for i0, piv0 in enumerate(num_pivot):
    
    temp = order0.copy()

    temp1 = temp[i0]
    temp[i0] = temp[piv0-1]
    temp[piv0-1] = temp1

    y = y[temp,:].copy()

print ('mat_P')
print (y)
print()

print ('mat_P_T')
print (y.T)
print()
print ('mat_p(scipy)')
print (p)
print()

print ('P.T L U')
print (y.T @ mat_L @ mat_U)
print()
print ('A')
print (mat_A)
