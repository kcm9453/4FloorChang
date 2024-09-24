import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()

# Dense Layer Class
class Dense:
    def __init__(self, n_input, n_output):
        self.weights = 0.01 * np.random.randn(n_input, n_output)
        self.biases = np.zeros((1, n_output))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output

# Activation ReLU Class
class ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
        return self.output

# Activation Softmax Class
class Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return self.output

# Loss_CategoricalCrossentropy Class
class CrossEntropy:
    def forward(self, prediction, targets):
        prediction = np.clip(prediction, 1e-7, 1 - 1e-7)
        if targets.ndim == 1:
            correct_confidences = prediction[np.arange(len(prediction)), targets]
        else:
            correct_confidences = np.sum(prediction * targets, axis=1)
        negative_log_likelihoods = -np.log(correct_confidences)
        return np.mean(negative_log_likelihoods)

# Create dataset
X, y = spiral_data(samples=100, classes=3)

# Forward pass
dense_layer = Dense(n_input=2, n_output=3)
relu = ReLU()
softmax = Softmax()
loss_function = CrossEntropy()

dense_output = dense_layer.forward(X)
relu_output = relu.forward(dense_output)
softmax_output = softmax.forward(relu_output)

# Loss calculation
loss = loss_function.forward(softmax_output, y)
print("Categorical Cross-Entropy Loss:", loss)