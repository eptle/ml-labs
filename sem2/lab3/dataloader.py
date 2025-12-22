import numpy as np

class DataLoader(object):
    def __init__(self, X, y, batch_size=1, shuffle=False):
        assert X.shape[0] == y.shape[0]
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_id = 0
        self.indices = np.arange(X.shape[0])

    def __len__(self):
        return (len(self.X) + self.batch_size - 1) // self.batch_size  

    def num_samples(self):
        return len(self.X)

    def __iter__(self):
        self.batch_id = 0
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self

    def __next__(self):
        if self.batch_id >= len(self.X):
            raise StopIteration    
        start_idx = self.batch_id
        end_idx = min(self.batch_id + self.batch_size, len(self.X))
        batch_indices = self.indices[start_idx:end_idx]
        x_batch = self.X[batch_indices]
        y_batch = self.y[batch_indices]
        self.batch_id += self.batch_size
        return x_batch, y_batch
