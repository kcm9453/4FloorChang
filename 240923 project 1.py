import matplotlib.pyplot as plt
import numpy as np
import nnfs
nnfs.init()

class Activation_SoftMax:

    def forward(self, inputs):
        return np.maximum(0,inputs)

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.uniform(0, 1, (n_inputs, n_neurons))
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output


Dense1 = Layer_Dense(1,8)
Dense2 = Layer_Dense(8,8)
Dense3 = Layer_Dense(8,1)

Dense1.weights = np.array([[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0]])
Dense1.biases = np.array([[0],[0],[0],[0],[0],[0],[0],[0]])

Dense2.weights = np.random.uniform(0,1,(3,3))
Dense2.biases = np.zeros([1,8])

X = np.linspace(0,2 * np.pi, 100).reshape(-1,1)
y = np.sin(X)

Activation = Activation_SoftMax

plt.plot(X, y,label="True Sine Wave", color="blue")
plt.plot(X, Activation.forward(X), label="NN Output", color="red")
plt.legend()
plt.title("Sine Wave Approximation using Neural Network")
plt.show()

dense1 = Layer_Dense(2, 3)

output = Activation.forward(Dense1.forward(x))
print(dense1.forward(X))
