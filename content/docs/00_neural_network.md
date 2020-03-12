---
title: "What is Neural Network?"
date: 2020-03-09
---	

## So what is a Neural Network?
---

It is a technique for building a computer program that can learn to perform some task by analyzing training data.

Artifical neural network was designed as a computational model based on our understanding of how human brain works.
An artifical neural network consists of number of (few to millions) simple processing nodes called *perceptrons* which are interconnected.

Lets have a quick look at a biological neuron before we take up perceptron (artifical neuron).

## Bilogical Neuron

![Bilogical Neuron](/images/c_00/biological_neuron.svg)

Biological neuron is the most basic computational unit of human brain.
An average human brain has approx 86 billion neurons. Each neuron may be connected to up to 10,000 other neurons.

Each neuron consists of
* *Dendrites* - takes care of receiving and processing incoming signals from other neurons
* *Axon* - produces output signal and connects to other neurons via synapses.

## Perceptron

An artificial neuron mimics computational model of a biological neuron. It is also the simplest neural network possible.

![Perceptron](/images/c_01/perceptron.png)

* A perceptron takes several inputs x1, x2, ... ,xn and generates a single output.
* The inputs are multiplied by the weights w1, w2, ... ,wn and added to create a weighted sum.
* A bias is added to weighted sum.
* An `activation function` is applied to the weighted sum to produce output for perceptron.

```python
# perceptron.py
import numpy as np
np.random.seed(4) # random seed for reproducible results
np.set_printoptions(precision=2) # pretty print numpy arrays

# Input
x = np.array([-1, 2, 1, 2]) # four inputs
w = np.random.randn(4) # four weights
b = 1 # bias

# activation step function
def activation_step(x):
    if x >= 1:
        return 1
    else:
        return 0

# a simple perceptron
def perceptron(x, w, b):
    weighted_sum = x.dot(w) + b
    print('Weighted sum:\n', weighted_sum)
    output = activation_step(weighted_sum)

    return output

print('Input:\n', x)
print('Weights:\n', w)
print('Bias:\n', b)

output = perceptron(x, w, b)
print('Output:\n', output)
```

Run the file 

```
#> python perceptron.py
Input:
 [-1  2  1  2]
Weights:
 [ 0.05  0.5  -1.    0.69]
Bias:
 1
Weighted sum:
 2.3406290448084768
Output:
 1
```

Hurray!!! we now have our first perceptron model in place. Simple isn't it?

Experiment with the above code a bit. 
What happens when you change the bias?

```python
x =  np.array([-1  2  1  2])
bias = -1 # changed from 1 to -1
```

```python
#> python perceptron.py
Input:
 [-1  2  1  2]
Weights:
 [ 0.05  0.5  -1.    0.69]
Bias:
 -1
Weighted sum:
 0.3406290448084768
Output:
 0
```

What happens if we change the input?

```python
x =  np.array([-1  2  1  2]) # changed from [-1  2  1  2] to [[-1  2  1  -2]]
bias = 1
```

```python
#> python perceptron.py
Input:
 [-1  2  1 -2]
Weights:
 [ 0.05  0.5  -1.    0.69]
Bias:
 1
Weighted sum:
 -0.4337649883567698
Output:
 0
```

Notice how the perceptron stops firing?

This simple perceptron can work as a binary classifier or we can stack many of them to form a *Multi-Layer Perceptron*.
Our artificial neural network is nothing but a multi-layer perceptron.

## A Neural Network

A neural network is stack of layered neurons which feed output of first layer to the next and so on.

![Multi-Layer perceptron](/images/c_01/neural_network_simple.png)

So a neural network has

1. **Input Layer:** The first layer of the network. This will correspond to number of inputs we have. Nothing much happens at this layer, the inputs are simply passed on to the next set of Hidden Layers.
2. **Hidden Layer:** This is where all the learning takes place. Learning process is nothing but constantly adjusting the weight values to see there is an improvement in the output.
Number of hidden layers and number of neurons / nodes per layer is something we will have to experiment with.\
If the hidden layer is too small our network will not learn anything useful resulting in *underfitting*. If the hidden layer is too large we would end up doing unnecessary calculations and *overfitting* the data.

3. **Output Layer:** The last layer of the network which is used to give predicted output. Number of neurons in this layer will depend on number of things being predicted.

----
## Reference:

* [Brain Neuron & Synapses](https://human-memory.net/brain-neurons-synapses/)
* [Overview of neuron structure and function](https://www.khanacademy.org/science/biology/human-biology/neuron-nervous-system/a/overview-of-neuron-structure-and-function)
* [Using neural nets to recognize handwritten digits](http://neuralnetworksanddeeplearning.com/index.html)
* [The Nature of Code - Chapter 10. Neural Networks](https://natureofcode.com/book/chapter-10-neural-networks/)
* [CS231n Convolutional Neural Networks for Visual Recognition](http://cs231n.github.io/neural-networks-1/)
* [The Human Brain in Numbers: A Linearly Scaled-up Primate Brain](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2776484/)