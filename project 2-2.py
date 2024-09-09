import numpy as np

inputs = [
        [1.0, 2.0, 3.0, 2.5],
        [2.0, 5.0, -1.0, 2.0],
        [-1.5, 2.7, 3.3, -0.8],
        [3.6, 2.4, 1.7, 5.3],
        [2.7, 5.3, -1.8, 2.6],
        ]
weights = [
        [0.2, 0.8, -0.5, 1.0],
        [0.5, -0.91, 0.26, -0.5],
        [-0.26, -0.27, 0.17, 0.87]
        ]

biases = [2.0,3.0,0.5]

layers_outputs = np.dot(inputs, np.array(weights).T) + biases

print(layers_outputs)
