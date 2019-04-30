import collections
import heapq
import numpy as np


class PrioritySet():
    # A priority queue without duplicate elements
    # credit assignment:
    # https://stackoverflow.com/questions/5997189/how-can-i-make-a-unique-value-priority-queue-in-python/5997409#5997409

    def __init__(self):
        self.heap = []
        self.set = set()

    def add(self, d, pri):
        if not d in self.set:
            heapq.heappush(self.heap, (pri, d))
            self.set.add(d)

    def pop(self):
        pri, d = heapq.heappop(self.heap)
        self.set.remove(d)
        return d

    def __len__(self):
        return len(self.heap)

    def __getitem__(self, key):
        return self.heap[key]


class MemoryBuffer():

    def __init__(self):

        self.max_size = 0
        self.memories = PrioritySet()

    def store_memory(self, x, y_target, loss):
        self.memories.add((x, y_target), loss)
        if len(self.memories) > self.max_size:
            self.memories.pop()

    def sample_memory(self):
        assert len(self.memories) > 0

        idx = np.random.randint(len(self.memories))
        _, (x_sample, y_target_sample) = self.memories[idx]
        return x_sample, y_target_sample

    def compute_loss_and_backward_pass_for_random_memory(self, model, criterion):
        x_sample, y_target_sample = self.sample_memory()
        y_sample = model(x_sample)
        loss_sample = criterion(y_sample, y_target_sample)

        self.store_memory(x_sample, y_target_sample, loss_sample.item())

        loss_sample.backward()


