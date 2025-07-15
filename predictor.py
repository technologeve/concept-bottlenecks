""" Predictor for CBM. """

# External imports
import torch.nn as nn


class Predictor(nn.Module):
    """ Predictor for CBM. """
    def __init__(self, n_concepts, latent_dims):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_concepts, latent_dims),
            nn.LeakyReLU(),
            nn.Linear(latent_dims, 1)
        )

    def forward(self, x):
        return self.net(x)
