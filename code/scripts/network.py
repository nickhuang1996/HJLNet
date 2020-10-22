# -*- coding: UTF-8 -*-
import numpy as np
from ..lib.Activation.Sigmoid import sigmoid_derivative, sigmoid
from ..lib.Module.Linear import Linear

class network:
    def __init__(self, layers_dim):
        self.layers_dim = layers_dim
        self.linear_list = [Linear(layers_dim[i - 1], layers_dim[i], bias=True) for i in range(1, len(layers_dim))]
        self.parameters = {}
        self._init_parameters()

    def _init_parameters(self):
        for i in range(len(self.layers_dim) - 1):
            self.parameters["w" + str(i)] = self.linear_list[i].weight
            self.parameters["b" + str(i)] = self.linear_list[i].bias

    def forward(self, x):
        a = []
        z = []
        caches = {}
        a.append(x)
        z.append(x)

        layers = len(self.parameters) // 2

        for i in range(layers):
            z_temp = self.linear_list[i].forward(a[i])
            self.parameters["w" + str(i)] = self.linear_list[i].weight
            self.parameters["b" + str(i)] = self.linear_list[i].bias
            z.append(z_temp)
            if i == layers - 1:
                a.append(z_temp)
            else:
                a.append(sigmoid(z_temp))
        caches["z"] = z
        caches["a"] = a
        return caches, a[layers]

    def backward(self, caches, output, y):
        layers = len(self.parameters) // 2
        grads = {}
        m = y.shape[1]

        for i in reversed(range(layers)):
            # 假设最后一层不经历激活函数
            # 就是按照上面的图片中的公式写的
            if i == layers - 1:
                grads["dz" + str(i)] = output - y
            else:  # 前面全部都是sigmoid激活
                grads["dz" + str(i)] = self.parameters["w" + str(i + 1)].T.dot(
                    grads["dz" + str(i + 1)]) * sigmoid_derivative(
                    caches["z"][i + 1])
            grads["dw" + str(i)] = grads["dz" + str(i)].dot(caches["a"][i].T) / m
            grads["db" + str(i)] = np.sum(grads["dz" + str(i)], axis=1, keepdims=True) / m
        return grads

    # 就是把其所有的权重以及偏执都更新一下
    def update_grads(self, grads, learning_rate):
        layers = len(self.parameters) // 2
        for i in range(layers):
            self.parameters["w" + str(i)] -= learning_rate * grads["dw" + str(i)]
            self.parameters["b" + str(i)] -= learning_rate * grads["db" + str(i)]






