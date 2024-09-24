import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()

def softmax(x):
    exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
    probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    return probabilities

class CrossEntropy:
    def forward(self, prediction, targets):
        prediction = np.clip(prediction, 1e-7, 1 - 1e-7)

        if targets.ndim == 1:
            correct_confidences = prediction[np.arange(len(prediction)), targets]
        else:
            correct_confidences = np.sum(prediction * targets, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return np.mean(negative_log_likelihoods)

X, y = spiral_data(samples=100, classes=3)

logits = np.array([
    [2.0, 1.0, 0.1],
    [0.1, 2.0, 1.0],
    [0.5, 0.2, 2.0]
])

softmax_outputs = softmax(logits)

# Cross-Entropy 손실 계산
loss_function = CrossEntropy()
loss = loss_function.forward(softmax_outputs, y)
print("Categorical Cross-Entropy Loss:", loss)