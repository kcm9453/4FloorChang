import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt

class DenseLayer:
    def __init__(self, input_dim, output_dim, initialize_method='xavier'):
        self.weights = self.initialize_weights(input_dim, output_dim, initialize_method)
        self.bias = np.zeros(output_dim)

    def initialize_weights(self, input_dim, output_dim, method):
        if method == 'xavier':
            return np.random.randn(input_dim, output_dim) * np.sqrt(1 / input_dim)
        elif method == 'he':
            return np.random.randn(input_dim, output_dim) * np.sqrt(2 / input_dim)
        elif method == 'gaussian':
            return np.random.randn(input_dim, output_dim) * 0.01
        else:
            return np.random.randn(input_dim, output_dim) * 0.01

    def forward(self, x):
        output = np.dot(x, self.weights) + self.bias
        output = np.maximum(0, output)
        return output


input_dim = 3
output_dim = 2


nnfs.init()
X, y = spiral_data(samples=100, classes=2)
plt.scatter(X[:, 0 ], X[:, 1 ], c = y, cmap = 'brg' )
plt.show()

methods = ['xavier', 'he', 'gaussian']

for method in methods:
    print(f"\n{method.capitalize()} 초기화 결과:")

    dense_layer = DenseLayer(input_dim, output_dim, initialize_method=method)

    input_data = np.array([[1.0, -1.0, 2.0], [3.0, 0.0, -2.0]])

    output_data = dense_layer.forward(input_data)

    print(output_data)