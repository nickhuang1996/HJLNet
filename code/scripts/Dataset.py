import numpy as np


class Dataset:
    def __init__(self):

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
