import numpy as np


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
