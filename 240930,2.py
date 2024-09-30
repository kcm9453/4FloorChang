import numpy as np
import nnfs
from nnfs.datasets import vertical_data

class Layer:
    def __init__(self, n_inputs, n_outputs):
        self.weights = 0.01 * np.random.randn(n_inputs, n_outputs)
        self.biases = np.zeros((1, n_outputs))

class Layer_Dense(Layer):
    def forward(self, inputs):
        return np.dot(inputs, self.weights) + self.biases

class Activation:
    def forward(self, inputs):
        raise NotImplementedError

class Activation_ReLU(Activation):
    def forward(self, inputs):
        return np.maximum(0, inputs)

class Activation_Softmax(Activation):
    def forward(self, inputs):
        exp = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        return np.mean(sample_losses)

class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        else:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        return -np.log(correct_confidences)

class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.loss_function = Loss_CategoricalCrossentropy()

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

class Trainer:
    def __init__(self, model, X, y):
        self.model = model
        self.X = X
        self.y = y
        self.lowest_loss = float('inf')

    def train(self, iterations):
        best_weights = [layer.weights.copy() for layer in self.model.layers]
        best_biases = [layer.biases.copy() for layer in self.model.layers]

        for iteration in range(iterations):
            for layer in self.model.layers:
                layer.weights += 0.05 * np.random.randn(*layer.weights.shape)
                layer.biases += 0.05 * np.random.randn(*layer.biases.shape)

            out = self.model.forward(self.X)
            loss = self.model.loss_function.calculate(out, self.y)

            if loss < self.lowest_loss:
                print(f"New best weights found at iteration {iteration}, loss: {loss}")
                best_weights = [layer.weights.copy() for layer in self.model.layers]
                best_biases = [layer.biases.copy() for layer in self.model.layers]
                self.lowest_loss = loss
            else:
                for i, layer in enumerate(self.model.layers):
                    layer.weights = best_weights[i]
                    layer.biases = best_biases[i]

X, y = vertical_data(samples=100, classes=3)
model = NeuralNetwork()
model.add(Layer_Dense(2, 3))
model.add(Activation_ReLU())
model.add(Layer_Dense(3, 3))
model.add(Activation_Softmax())

trainer = Trainer(model, X, y)
trainer.train(100000)