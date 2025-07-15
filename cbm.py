""" Encoder for CBM. """

# External imports
import torch.nn as nn
from torch_concepts.nn import LinearConceptLayer

# Internal imports
from encoder import Encoder
from predictor import Predictor


class ConceptLayer(nn.Module):
    """ Concept Layer for CBM. """
    def __init__(self, latent_dims, concept_names):
        super().__init__()
        self.c_layer = LinearConceptLayer(
        in_features=latent_dims,
        out_annotations=concept_names
    )
    def forward(self, x):
        return self.c_layer(x)


class CBM(nn.Module):
    """ Encoder for CBM. """
    def __init__(self, n_concepts, latent_dims, concept_names):
        super().__init__()
        self.encoder = Encoder(latent_dims)
        self.c_layer = ConceptLayer(latent_dims, concept_names)
        self.predictor = Predictor(n_concepts, latent_dims)

    def forward(self, x):
        enc = self.encoder(x)
        lay = self.c_layer(enc).sigmoid()
        pred = self.predictor(lay).sigmoid().view(-1)
        return {"enc": enc, "lay": lay, "pred": pred}
