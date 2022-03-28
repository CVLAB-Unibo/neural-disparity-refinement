import torch
import torch.nn as nn

class Sine(nn.Module):
    def __init__(self, w0: float = 1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x: torch.Tensor):
        return torch.sin(self.w0 * x)
