import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weight = np.random.uniform(0,1,(n_inputs, n_neurons))
        self.bias = np.random.uniform(0,1,(1, n_neurons))

    def forward(self, inputs):
        return np.dot(inputs,self.weight) + self.bias

nnfs.init()
X, y = spiral_data(samples=100, classes=2)
plt.scatter(X[:, 0 ], X[:, 1 ], c = y, cmap = 'brg' )
plt.show()


