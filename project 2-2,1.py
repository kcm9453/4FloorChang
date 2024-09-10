import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt

class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def f_print(self):
        print("My name is {0}".format(self.name))
        print("My age is {0}".format(self.age))

        p1 = Person("홍길동", 30)
        p2 = Person("이순신", 20)

nnfs.init()
    X, y = spiral_data(samples=100, classes=3)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg')
    plt.show()