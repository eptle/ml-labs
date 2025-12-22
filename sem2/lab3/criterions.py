import numpy as np
from base import Criterion
from activations import LogSoftmax

class MSELoss(Criterion):
    def compute_output(self, input, target):
        assert input.shape == target.shape, 'input and target shapes not matching'
        return np.square(input - target).mean()

    def compute_grad_input(self, input, target):
        assert input.shape == target.shape, 'input and target shapes not matching'
        return 2 * (input - target) / input.size

class CrossEntropyLoss(Criterion):
    def __init__(self):
        super().__init__()
        self.log_softmax = LogSoftmax()

    def compute_output(self, input, target):
        batch_size = input.shape[0]
        log_probs = self.log_softmax.compute_output(input)
        target_log_probs = log_probs[np.arange(batch_size), target]
        loss = -np.mean(target_log_probs)
        return loss

    def compute_grad_input(self, input, target):
        batch_size = input.shape[0]
        num_classes = input.shape[1]
        log_probs = self.log_softmax.compute_output(input)
        probs = np.exp(log_probs)
        one_hot_targets = np.zeros((batch_size, num_classes))
        one_hot_targets[np.arange(batch_size), target] = 1
        grad = (probs - one_hot_targets) / batch_size
        return grad
