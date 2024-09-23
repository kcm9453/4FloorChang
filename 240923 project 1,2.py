import matplotlib.pyplot as plt
import numpy as np
import nnfs
nnfs.init()

# 활성화 함수: Tanh
class Activation_Tanh:
    def forward(self, inputs):
        # Tanh 활성화 함수 적용
        return np.tanh(inputs)

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # 가중치 및 편향 초기화 (작은 값으로 설정하여 훈련이 잘 되도록 함)
        self.weights = np.random.randn(n_inputs, n_neurons) * 0.1
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        # Forward 계산: 가중치와 편향 적용
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output

# 입력 데이터: 사인파 생성
X = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)
y = np.sin(X)

# 신경망 레이어 설정 (8개의 뉴런을 가진 두 개의 레이어)
Dense1 = Layer_Dense(1, 8)  # 첫 번째 레이어: 입력 1개, 뉴런 8개
Dense2 = Layer_Dense(8, 8)  # 두 번째 레이어: 입력 8개, 뉴런 8개
Dense3 = Layer_Dense(8, 1)  # 출력 레이어: 입력 8개, 출력 1개

# 활성화 함수 설정
activation1 = Activation_Tanh()
activation2 = Activation_Tanh()

# Forward propagation
dense1_output = Dense1.forward(X)
activation1_output = activation1.forward(dense1_output)

dense2_output = Dense2.forward(activation1_output)
activation2_output = activation2.forward(dense2_output)

dense3_output = Dense3.forward(activation2_output)

# 결과 시각화
plt.plot(X, y, label="True Sine Wave", color="blue")
plt.plot(X, dense3_output, label="NN Output", color="red")
plt.legend()
plt.title("Tanh 활성화를 사용한 신경망의 사인파 근사")
plt.show()

# 가중치와 편향 출력
print("Dense1 Weights:\n", Dense1.weights)
print("Dense1 Biases:\n", Dense1.biases)
print("Dense2 Weights:\n", Dense2.weights)
print("Dense2 Biases:\n", Dense2.biases)
print("Dense3 Weights:\n", Dense3.weights)
print("Dense3 Biases:\n", Dense3.biases)