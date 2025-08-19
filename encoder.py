""" Encoder for CBM. """

# External imports
import torch.nn as nn

class Encoder(nn.Module):
    """ Encoder for CBM. """
    def __init__(self, latent_dims, channels_in=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels_in, 4, (3, 3), padding='same'),
            nn.LeakyReLU(),

            nn.Conv2d(4, 4, (3, 3), padding='same'),
            nn.LeakyReLU(),
            nn.BatchNorm2d(4),

            nn.Conv2d(4, 4, (3, 3), padding='same'),
            nn.LeakyReLU(),

            nn.Conv2d(4, 4, (3, 3), padding='same'),
            nn.LeakyReLU(),
            nn.BatchNorm2d(4),

            nn.MaxPool2d((5, 5)),

            nn.Flatten(start_dim=1, end_dim=-1),
            nn.LazyLinear(latent_dims),  
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.net(x)
