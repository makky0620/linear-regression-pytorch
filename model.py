# coding: utf-8

import torch.nn as nn

class LinerRegression(nn.Module):

    def __init__(self, input_size):
        super(LinerRegression, self).__init__()
        self.fc1 = nn.Linear(input_size, 1)
    
    def forward(self, input):
        output = self.fc1(input)
        return output
