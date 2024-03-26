from ..lib.Data.DataLoader import DataLoader
from ..scripts.Dataset import Dataset
from ..scripts.network import network
import matplotlib.pyplot as plt
import numpy as np


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
        plt.savefig(fname=self.result_img_path)

    # �������ֵ
    def compute_loss(self, output, y):
        return np.mean(np.square(output - y))
