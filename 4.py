import numpy as np

inputs = [-1, 0, 1]
num_neuron = 3
outputs = cal_neuron(num_neuron, inputs)

def init_weight(inputs):

    weights = np.random.uniform(-1, 1, size=len(inputs))
    return weights

def cal_neuron(num_neuron, inputs):
    outputs = []

    for _ in range(num_neuron):
        weights = init_weight(inputs)
        neuron_output = np.dot(weights, inputs)
        outputs.append(neuron_output)

    return outputs
