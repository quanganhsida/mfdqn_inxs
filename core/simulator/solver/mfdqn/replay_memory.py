from collections import namedtuple, deque
import random

Transition = namedtuple('Transition',
                        ('observation', 'mf_action', 'action', 'next_observation', 'next_mf_action', 'reward'))

class ReplayMemory:

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
