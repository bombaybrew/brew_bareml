---
title: "Linear Algebra - Basics"
date: 2020-02-01
---

# Linear Algebra
---
Status: WIP

In this chapter we are going to cover what is a Tensor.
We will also see how to code a tensor in python.

## Why is this important?

For machine learning input data is represented as one or multi dimentional Tensor. Knowing how to efficiently transform and operate on this data becomes important.

Simple understanding of these operations will make it easier of us to read through source code of various ML libraries.

Let's understand Scalar, Vector & Matrix

### Scalar
Scalar is a just a number.

```python
a = 20  # int
b = 0.1 # float
```

### Vector
Multipe numbers in a array and we have a vector

```python
import numpy as np

vector = np.array([1, 2, 3])
print(vector)

# ----- Output
[1 2 3]
```

### Matrix
Matrix is 2 dimentional array of numbers

```python
import numpy as np

matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(matrix)

# ----- Output
[[1 2 3]
 [4 5 6]
 [7 8 9]]
```
---

## Tensor

A Tensor is nothing but generalisation of vector & matrices in higher dimentions.

So a vector is one dimentional tensor and a matrix is a 2 dimentional vector.


```python
import numpy as np

# Create tensor a and b

a = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
print('Tensor A \n', a)

b = np.array([[2, 2, 2], [4, 4, 4], [6, 6, 6]])
print('\nTensor B \n', b)

# ----- Output
Tensor A 
 [[1 1 1]
 [2 2 2]
 [3 3 3]]

Tensor B 
 [[2 2 2]
 [4 4 4]
 [6 6 6]]
```

Tensor basic operations

```python
# Addition
add = np.add(a, b)
print('\nAddition \n', add)

# Subtraction
sub = np.subtract(a, b)
print('\nSubtraction \n', sub)

# Multiplication / Dot Product
dot = np.dot(a, b)
print('\nDot product \n', dot)

# Transpose
a_transpose = np.transpose(a)
print('\nA Transpose \n', a_transpose)

# ----- Output

Addition 
 [[3 3 3]
 [6 6 6]
 [9 9 9]]

Subtraction 
 [[-1 -1 -1]
 [-2 -2 -2]
 [-3 -3 -3]]

Dot product 
 [[12 12 12]
 [24 24 24]
 [36 36 36]]

A Transpose 
 [[1 2 3]
 [1 2 3]
 [1 2 3]]
```

----
## Reference:

* [Linear Algebra - Numpy](https://docs.scipy.org/doc/numpy/reference/routines.linalg.html)





