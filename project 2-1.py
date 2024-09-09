import numpy as np

inputs = [
        [1.4, 2.4, 3.4, 2.9],
        [2.4, 5.4, -1.4, 2.4],
        [-1.9, 3.1, 3.7, -1.2]
        ]
weights = [
        [0.8, 1.4, -1.1, 1.6],
        [1.1, -1.51, 0.86, -1.1],
        [-0.86, -0.87, 0.77, 1.47]
        ]

biases = [2.0,3.0,0.5]

layers_outputs = np.dot(inputs, np.array(weights).T) + biases

print(layers_outputs)
