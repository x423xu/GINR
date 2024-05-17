import torch
import torch.nn as nn
from .siren import Siren

class Siren_decoder(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.C = 1
        self.decoder = Siren(args=args, in_features=3, out_features=self.C, outermost_linear=True)
        if args.hidden_dim != 256:
            self.latent_embedding = nn.Linear(1024, args.hidden_dim*args.num_layers, bias = False)
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
    
    def forward(self, coords, z, return_latent = False):
        if self.hidden_dim != 256:
            z = self.latent_embedding(z)
        pos_embedding = self.decoder.map(coords)
        ac0 = self.decoder.net.net[0](pos_embedding)
        o1 = self.decoder.net.net[1][0](ac0)
        o1 = o1 + z[:,:self.hidden_dim].unsqueeze(1)
        ac1 = self.decoder.net.net[1][1](o1)
        o=ac1
        for i in range(2, self.num_layers+1):
            o = self.decoder.net.net[i][0](o)
            o = o + z[:,self.hidden_dim*(i-1):self.hidden_dim*i].unsqueeze(1)
            o = self.decoder.net.net[i][1](o)
        o = self.decoder.net.net[-1](o)
        if return_latent:
            return o, z
        else:
            return o, None
    def get_parameters(self):
        return list(self.decoder.parameters()) + list(self.latent_embedding.parameters())