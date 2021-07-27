# cuBLASTutorial
This project aims to understand how to use the cuBLAS library.


## example 1: cublas PA=LU decomposition
let's begin by introducing a random matrix A 
```python
A = 
[[0.065583 0.109075 0.886625 0.552285 0.985441]
 [0.77063  0.874429 0.335947 0.747717 0.645174]
 [0.461059 0.394548 0.831258 0.601085 0.933953]
 [0.728199 0.317596 0.369686 0.084441 0.164917]
 [0.543888 0.235383 0.79529  0.206442 0.06706 ]]
```
the decomposition function looks as following:
```CUDA C
cublasStatus_t cublasSgetrfBatched(cublasHandle_t handle,
                                   int n,
                                   float *const Aarray[],
                                   int lda,
                                   int *PivotArray,
                                   int *infoArray,
                                   int batchSize);
```
To do the LU decomposition, the matrix should be stored inside GPU memory, which is colomn-major because of cuBLAS.
the pivotArray [M], where M is the size of the matrix in x and in y; the infoArray[batchSize] stores the return information (success or failed).

Basically, the **cublasSgetrfBatched** function do the PA=LU decomposition. In scipy, the function **scipy.linalg.lu** returns inverse permutation matrix **P.T**. In other words, scipy is doing **A = p l u**, where **p** is the **P.T** mentioned above.

after calling the function, the LU are stored inplace of Aarray. 
```python
Aarray = 
[[0.065583 0.109075 0.886625 0.552285 0.985441]
 [0.77063  0.874429 0.335947 0.747717 0.645174]
 [0.461059 0.394548 0.831258 0.601085 0.933953]
 [0.728199 0.317596 0.369686 0.084441 0.164917]
 [0.543888 0.235383 0.79529  0.206442 0.06706 ]]
```

the L matrix can be reconstructed by 
```python
L[i,j] = Aarray[i,j], if i>j
L[i,j] = 1, if i==j
L[i,j] = 0, else
```
the U matrix can be reconstructed by 
```python
L[i,j] = Aarray[i,j], if i<=j
L[i,j] = 0, else
```
the permutation matrix can be reconstructed from eye matrix with pivotArray
```python
x = np.eye(5)
for i, piv in enumerate(pivotArray):
    swap row i and row piv of x
```


```python
mat_P = 
[[0. 1. 0. 0. 0.]
 [0. 0. 0. 1. 0.]
 [1. 0. 0. 0. 0.]
 [0. 0. 0. 0. 1.]
 [0. 0. 1. 0. 0.]]
 
 mat_P_T = mat_P.T
[[0. 0. 1. 0. 0.]
 [1. 0. 0. 0. 0.]
 [0. 0. 0. 0. 1.]
 [0. 1. 0. 0. 0.]
 [0. 0. 0. 1. 0.]]
```

```python
mat_L = 
[[ 1.        0.        0.        0.        0.      ]
 [ 0.94494   1.        0.        0.        0.      ]
 [ 0.085103 -0.068132  1.        0.        0.      ]
 [ 0.705771  0.750489  0.602356  1.        0.      ]
 [ 0.598288  0.252833  0.716182  0.069673  1.      ]]

mat_U = 
[[ 0.77063   0.874429  0.335947  0.747717  0.645174]
 [ 0.       -0.508686  0.052236 -0.622107 -0.444734]
 [ 0.        0.        0.861593  0.446267  0.900234]
 [ 0.        0.        0.       -0.123202 -0.596779]
 [ 0.        0.        0.        0.        0.057244]]
```

so that, **A = P.T L U** is fulfilled.
```python
results = 
[[0.06558292 0.10907433 0.88662415 0.55228535 0.98544086]
 [0.77063    0.874429   0.335947   0.747717   0.645174  ]
 [0.46105868 0.39454777 0.83125744 0.60108547 0.93395243]
 [0.72819911 0.31759694 0.36968576 0.0844407  0.16491672]
 [0.54388831 0.23538338 0.79528991 0.20644212 0.06705948]]
```
