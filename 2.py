import random

def init_weights(inputs):
    weights = []
    for i in range(len(inputs)):
            weights.append(random.uniform(-1,1))
    return weights