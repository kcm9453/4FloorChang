import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()

class cross_entropy:
    def forward(self, prediction, targets):
        prediction = np.clip(prediction, 1e-7, 1 - 1e-7)

        if targets.ndim == 1:
            correct_confidences = prediction[np.arange(len(prediction)), targets]
        else:
            correct_confidences = np.sum(prediction * targets, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return np.mean(negative_log_likelihoods)

def categorical_cross_entropy(prediction,targets):
    prediction = np.clip(prediction, 1e-7, 1 - 1e-7)

    if targets.ndim == 1:
        correct_confidences = prediction[np.arange(len(prediction)), targets]
    else:
        correct_confidences = np.sum(prediction * targets, axis=1)

    negative_log_likelihoods = -np.log(correct_confidences)
    return np.mean(negative_log_likelihoods)


softmax_outputs = np.array([
    [0.7, 0.1, 0.2],
    [0.1, 0.5, 0.4],
    [0.2, 0.2, 0.6]
])
targets = np.array([0,1,2])

loss = categorical_cross_entropy(softmax_outputs, targets)
print("Categorical Cross-Entropy Loss", loss)

X, y = spiral_data(samples=100, classes=3)