import numpy as np
import random
import torch

from memory_buffer import MemoryBuffer


class Net(torch.nn.Module):

    def __init__(self, in_features, n_hidden, out_features):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.f_internal = torch.relu
        self.f_output = lambda x: x

        self.layers = []
        for i in range(len(n_hidden) + 1):
            if i == 0:
                inf = in_features
            else:
                inf = n_hidden[i - 1]
            if i == len(n_hidden):
                outf = out_features
            else:
                outf = n_hidden[i]

            self.layers.append(torch.nn.Linear(inf, outf))
            self.add_module(f'layer{i}', self.layers[i])

        self.memory_buffer = MemoryBuffer()
        self.use_memory_buffer = False

    @property
    def n_layers(self):
        return len(self.layers)

    def forward(self, x):
        for i in range(self.n_layers - 1):
            x = self.f_internal(self.layers[i](x))

        return self.f_output(self.layers[-1](x))

    def clone_parameters(self, other):
        for i in range(self.n_layers):
            self.layers[i].weight.data = other.layers[i].weight.data.clone()
            self.layers[i].bias.data = other.layers[i].bias.data.clone()
