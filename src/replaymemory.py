from collections import deque
from random import sample


class ReplayMemory():
    def __init__(self, max_size, minibatch_size):
        self.buffer = deque(maxlen=max_size)
        self.minibatch_size = minibatch_size

    def add(self, state, action, reward, done, next_state):
        self.buffer.append((state, action, reward, done, next_state))

    def sample(self):
        try:
            return sample(self.buffer, self.minibatch_size)
        except ValueError:
            return self.buffer
