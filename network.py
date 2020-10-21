# -*- coding: UTF-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from easydict import EasyDict

_C = EasyDict()
_C.layers_dim = [1, 25, 1]
_C.batch_size = 90
_C.total_epochs = 40000
_C.resume = True  # False means retraining
_C.result_img_path = "result.png"
_C.ckpt_path = "ckpt.npy"

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


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


class Dataset:
    def __init__(self):
        """
        加载数据集
        """
        self.x = np.arange(0.0, 2.0, 0.01)
        self.y = 20 * np.sin(2 * np.pi * self.x)
        self.length = len(list(self.x))
        self._build_items()
        self._transform()

    def _build_items(self):
        self.items = [{
            'x': list(self.x)[i],
            'y': list(self.y)[i]
        }for i in range(self.length)]

    def _transform(self):
        self.x = self.x.reshape(1, self.__len__())
        self.y = self.y.reshape(1, self.__len__())

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.items[index]


class DataLoader:
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.current = 0

    def __next__(self):
        if self.current < len(self.dataset.items):
            if self.current + self.batch_size <= len(self.dataset.items):
                item = self._concate([self.dataset.__getitem__(index) for index in range(self.current, self.current + self.batch_size)])
                # item = self._concate(self.dataset.items[self.current: self.current + self.batch_size])
                self.current += self.batch_size
            else:
                item = self._concate([self.dataset.__getitem__(index) for index in range(self.current, len(self.dataset.items) - 1)])
                # item = self._concate(self.dataset.items[self.current: len(self.dataset.items) - 1])
                self.current = len(self.dataset.items)
            return item
        else:
            self.current = 0
            raise StopIteration

    def _concate(self, dataset_items):
        concated_item = {}
        for item in dataset_items:
            for k, v in item.items():
                if k not in concated_item:
                    concated_item[k] = [v]
                else:
                    concated_item[k].append(v)
        concated_item = self._transform(concated_item)
        return concated_item

    def _transform(self, concated_item):
        for k, v in concated_item.items():
            concated_item[k] = np.array(v).reshape(1, len(v))
        return concated_item

    def __iter__(self):
        return self

class Trainer:
    def __init__(self, cfg):
        self.ckpt_path = cfg.ckpt_path
        self.result_img_path = cfg.result_img_path
        self.layers_dim = cfg.layers_dim
        self.net = network(self.layers_dim)
        if cfg.resume:
            self.load_models()
        self.dataset = Dataset()
        self.dataloader = DataLoader(dataset=self.dataset, batch_size=cfg.batch_size)
        self.total_epochs = cfg.total_epochs
        self.iterations = 0
        self.x = self.dataset.x
        self.y = self.dataset.y
        self.draw_data(self.x, self.y)

    def train(self):
        for i in range(self.total_epochs):

            for item in self.dataloader:
                caches, output = self.net.forward(item['x'])
                grads = self.net.backward(caches, output, item['y'])
                self.net.update_grads(grads, learning_rate=0.03)
                if i % 100 == 0:
                    print("Epoch: {}/{} Iteration: {} Loss: {}".format(i + 1,
                                                                       self.total_epochs,
                                                                       self.iterations,
                                                                       self.compute_loss(output, item['y'])))
                self.iterations += 1

    def test(self):
        caches, output = self.net.forward(self.x)
        self.draw_data(self.x, output)
        self.save_results()
        self.show()

    def save_models(self):
        ckpt = {
            "layers_dim": self.net.layers_dim,
            "parameters": self.net.linear_list
        }
        np.save(self.ckpt_path, ckpt)
        print('Save models finish!!')

    def load_models(self):
        ckpt = np.load(self.ckpt_path).item()
        self.net.layers_dim = ckpt["layers_dim"]
        self.net.linear_list = ckpt["parameters"]
        print('load models finish!!')

    def draw_data(self, x, y):
        plt.scatter(x, y)

    def show(self):
        plt.show()

    def save_results(self):
        plt.savefig(fname=self.result_img_path, figsize=[10, 10])

    # 计算误差值
    def compute_loss(self, output, y):
        return np.mean(np.square(output - y))


if __name__ == '__main__':
    trainer = Trainer(cfg=_C)
    trainer.train()
    trainer.test()
    trainer.save_models()




