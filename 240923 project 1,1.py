import matplotlib.pyplot as plt
import numpy as np
import nnfs
nnfs.init()

class Activation_SoftMax:
    def forward(self, inputs):
        # 소프트맥스 함수 적용
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))  # 안정성 위한 계산
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return probabilities

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # 가중치는 랜덤 값으로, 편향은 0으로 초기화
        self.weights = np.random.uniform(0, 1, (n_inputs, n_neurons))
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        # Forward 계산
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output


# 레이어와 활성화 함수 설정
Dense1 = Layer_Dense(1, 8)  # 입력 레이어: 1개의 입력과 8개의 뉴런
Dense2 = Layer_Dense(8, 8)  # 히든 레이어: 8개의 뉴런과 8개의 출력
Dense3 = Layer_Dense(8, 1)  # 출력 레이어: 1개의 출력 뉴런

# 입력 및 출력 데이터 (사인파)
X = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)
y = np.sin(X)

# 레이어를 통한 Forward Propagation
dense1_output = Dense1.forward(X)
dense2_output = Dense2.forward(dense1_output)
dense3_output = Dense3.forward(dense2_output)

# 결과 시각화
plt.plot(X, y, label="True Sine Wave", color="blue")
plt.plot(X, dense3_output, label="NN Output", color="red")
plt.legend()
plt.title("신경망을 사용한 사인파 근사")
plt.show()