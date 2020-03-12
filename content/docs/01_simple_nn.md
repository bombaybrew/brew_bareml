---
title: "Building a simple Neural Network"
date: 2020-02-08
---

# A Simple Neural Network
---
Status: WIP

## Building [XNOR gate](https://en.wikipedia.org/wiki/XNOR_gate) using a simple neural network

Truth table for XNOR gate is as follows. A and B being the inputs.

|A  | B | A XNOR B |
|---|----|---|
|0  | 0  | 1 |
|0  | 1  | 0 |
|1  | 0  | 0 |
|1  | 1  | 1 |	


We are going to build a neural network which has
1. **Input Layer:** The first layer of the network with 2 nodes.
2. **Hidden Layer:** To keep things simple, for this experiment we are going to use only one hidden layer and use only 4 nodes for this layer. It will be easier for us to print and follow the network configuration for this toy network we are building.
3. **Output Layer:** A single node layer which should match (A XNOR B) output.

It will look something like this
![Neural Network for XNOR](/images/c_01/neural_network_simple.png)

Code for this experiment is available at this [git repo.](https://github.com/bombaybrew/brew_nlp)

## Imports, Constants and Inputs

We will only need numpy library ('pip install numpy' if you don't already have it).

Let's create a new file 'XNOR_NN.py' and setup the following.

```python
import numpy as np
np.random.seed(2)
np.set_printoptions(precision=2)
```

Set random seed for numpy so that we can reproduce the same results for various runs.
Set printoptions so that when we print arrays values are readable.


```python
# Constants
CONST_epocs = 500
CONST_learning_rate = 0.001
CONST_print_log = True

CONST_H1_DIMEN = 4

# Input
x =  np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# Output
y = np.array([[1], [0], [0], [1]])

# Weights
w1 = np.random.random((2,CONST_H1_DIMEN))
w2 = np.random.random((CONST_H1_DIMEN, 1))

print("INPUT:\n", x)
print("OUTPUT:\n", y)

print("WEIGHTS L1:\n", w1)
print("WEIGHTS L2:\n", w2)
```

Next up we have constants that we can play with to experiment quickly.
Input numpy array 'x' and output array 'y' reflect our truth table for XNOR function.

Let's run `python XNOR_NN.py`

```python
INPUT:
 [[0 0]
 [0 1]
 [1 0]
 [1 1]]
OUTPUT:
 [[1]
 [0]
 [0]
 [1]]
WEIGHTS HIDDEN [w1]:
 [[0.44 0.03 0.55 0.44]
 [0.42 0.33 0.2  0.62]]
WEIGHTS OUTPUT [w2]:
 [[0.3 ]
 [0.27]
 [0.62]
 [0.53]]
```

Running our code nicely prints all the input, output and weights.


## Forward Propogation

Once we have Input (x), Output (y) and weights (w1 for hidden layer, w2 for output layer) we will now attempt a forward pass.

```python
# Train Network
for epoch in range(CONST_epocs):

    # forward pass
    h = x.dot(w1)
    h_activation = np.maximum(h, 0)
    y_pred = h_activation.dot(w2)

    # print forward pass
    if CONST_print_log:
        print("\n#---- EPOCH: ", epoch)
        print("Output Hidden:\n", h)
        print("Output Hidden + activation fn:\n", h_activation)
        print("Output L2 / y_pred:\n", y_pred)
```

An epoch is complete pass through entire training data set. In our case 'x' denotes our training set with 4 records.

**Step_1:** Perform `x.dot(w1)` which does following

![forward pass calculation](/images/c_01/nn_forward_pass_x_w1.png)

**Step_2:** Apply an activation function which is RELU in our case. This gives us `h_activation`
Read more about [RELU here.](https://www.tinymind.com/learn/terms/relu)

**Step_3:** Perform `h_activation.dot(w2)` which will give us our output or 'y_pred'

Lets run our code. We can verify the intermediate calculations done by forward pass.

```python
INPUT:
 [[0 0]
 [0 1]
 [1 0]
 [1 1]]
OUTPUT:
 [[1]
 [0]
 [0]
 [1]]
WEIGHTS HIDDEN [w1]:
 [[0.44 0.03 0.55 0.44]
 [0.42 0.33 0.2  0.62]]
WEIGHTS OUTPUT [w2]:
 [[0.3 ]
 [0.27]
 [0.62]
 [0.53]]

#---- EPOCH:  0
Output Hidden:
 [[0.   0.   0.   0.  ]
 [0.42 0.33 0.2  0.62]
 [0.44 0.03 0.55 0.44]
 [0.86 0.36 0.75 1.05]]
Output Hidden + activation fn:
 [[0.   0.   0.   0.  ]
 [0.42 0.33 0.2  0.62]
 [0.44 0.03 0.55 0.44]
 [0.86 0.36 0.75 1.05]]
Output L2 / y_pred:
 [[0.  ]
 [0.67]
 [0.71]
 [1.38]]

#---- EPOCH:  1
Output Hidden:
 [[0.   0.   0.   0.  ]
 [0.42 0.33 0.2  0.62]
 [0.44 0.03 0.55 0.44]
 [0.86 0.36 0.75 1.05]]
Output Hidden + activation fn:
 [[0.   0.   0.   0.  ]
 [0.42 0.33 0.2  0.62]
 [0.44 0.03 0.55 0.44]
 [0.86 0.36 0.75 1.05]]
Output L2 / y_pred:
 [[0.  ]
 [0.67]
 [0.71]
 [1.38]]

#---- EPOCH:  2
Output Hidden:
 [[0.   0.   0.   0.  ]
 [0.42 0.33 0.2  0.62]
 [0.44 0.03 0.55 0.44]
 [0.86 0.36 0.75 1.05]]
Output Hidden + activation fn:
 [[0.   0.   0.   0.  ]
 [0.42 0.33 0.2  0.62]
 [0.44 0.03 0.55 0.44]
 [0.86 0.36 0.75 1.05]]
Output L2 / y_pred:
 [[0.  ]
 [0.67]
 [0.71]
 [1.38]]
```