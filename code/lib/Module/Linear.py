import numpy as np


class Linear:
    def __init__(self, in_features, out_features, bias=False):
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self._init_parameters()

    def _init_parameters(self):
        self.weight = np.random.random([self.out_features, self.in_features])
        if self.bias:
            self.bias = np.zeros([self.out_features, 1])
        else:
            self.bias = None

    def forward(self, input):
        return self.weight.dot(input) + self.bias
