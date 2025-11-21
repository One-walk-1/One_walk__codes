import torch
from torch import nn
class PositionalEncoder(nn.Module):
    r"""
    Borrowed from: Mason McGough 
    Description: Sine-cosine positional encoder for input points.
    Source: https://towardsdatascience.com/its-nerf-from-nothing-build-a-vanilla-nerf-with-pytorch-7846e4c45666
    """
    def __init__(
      self,
      d_input: int,
      n_freqs: int,
      log_space: bool = False
    ):
        super().__init__()
        self.d_input = d_input
        self.n_freqs = n_freqs
        self.log_space = log_space
        self.d_output = d_input * (1 + 2 * self.n_freqs)
        self.embed_fns = [lambda x: x]

        # Define frequencies in either linear or log scale
        if self.log_space:
            freq_bands = 2.**torch.linspace(0., self.n_freqs - 1, self.n_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**(self.n_freqs - 1), self.n_freqs)

        # Alternate sin and cos
        for freq in freq_bands:
            self.embed_fns.append(lambda x, freq=freq: torch.sin(x * freq))
            self.embed_fns.append(lambda x, freq=freq: torch.cos(x * freq))
  
    def __repr__(self):
        return super().__repr__()+ f": d_input: {self.d_input}, d_output: {self.d_output}"

    def forward(
        self,
        x
    ) -> torch.Tensor:
        r"""
        Apply positional encoding to input.
        """
        return torch.concat([fn(x) for fn in self.embed_fns], dim=-1)

import torch

class FreqGate:
    def __init__(self, L:int, start_step:int, end_step:int):
        assert L >= 1 and end_step > start_step
        self.L = L
        self.start = int(start_step)
        self.end = int(end_step)

    def weights(self, global_step:int) -> torch.Tensor:
        if global_step <= self.start:
            k_active = 1
        elif global_step >= self.end:
            k_active = self.L
        else:
            ratio = (global_step - self.start) / float(self.end - self.start)
            k_active = max(1, int(round(1 + ratio * (self.L - 1))))
        w = torch.zeros(self.L, dtype=torch.float32)
        w[:k_active] = 1.0
        return w