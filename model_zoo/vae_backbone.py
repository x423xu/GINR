'''
Here we explore different backbones for inferring params: vae decoder tailored for 1d features.
'''
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat

'''
For vae_backbone, we need to specify:
    1. how many layers for each encoder/decoder cell.
    2. how to be compatible with 1d features.

For a simple vae for 20 pointclouds data, we only use z (b, 128), feat (b, 1024) to generate new samples and fit kld.
To scale up to large size data like 100K samples, we need to consider:
    1. If we should increase z's size to express more data.
    2. If we should increase feat's size so as to increase siren's capacity.
As the first attempt, we will keep z's and feat's sizes fixed. We increase the encoder and decoder number of cell to implement a hierarchical structure.
Secondly, we will try to increase z's size to express more data.
Thirdly, we will try to increase feat's size to increase siren's capacity.
What is more, we have to be very careful for the sigma of prior distribution to avoid under/over-fitting to prior/data.
'''

#--------------------Component classes--------------------#
# Encoder combiner cell
class EncCombinerCell(nn.Module):
    def __init__(self, Cin2, Cout):
        super(EncCombinerCell, self).__init__()
        # Cin = Cin1 + Cin2
        self.conv = nn.Conv1d(Cin2, Cout, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x1, x2):
        x2 = self.conv(x2)
        out = x1 + x2
        return out

#Decoder combiner cell
class DecCombinerCell(nn.Module):
    def __init__(self, Cin1, Cin2, Cout):
        super(DecCombinerCell, self).__init__()
        self.conv = nn.Conv1d(Cin2+Cin1, Cout, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x1, x2):
        out = torch.cat([x1, x2], dim=1)
        out = self.conv(out)
        return out

class ZCombiner(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.combiner = nn.Sequential(
                                        nn.Linear(inc, outc),
                                        nn.SELU(),
                                        nn.Linear(outc, outc),
                                        )
    
    def forward(self, z):
        s = self.combiner(z)
        mu, log_sigma = torch.chunk(s, 2, dim=2)

        return mu, log_sigma

@torch.jit.script
def soft_clamp5(x: torch.Tensor):
    return x.div(5.).tanh_().mul(5.) 

@torch.jit.script
def sample_normal_jit(mu, sigma):
    eps = mu.mul(0).normal_()
    z = eps.mul_(sigma).add_(mu)
    return z, eps
class Normal:
    def __init__(self, mu, log_sigma, temp=1.):
        self.mu = soft_clamp5(mu)
        log_sigma = soft_clamp5(log_sigma)
        self.sigma = torch.exp(log_sigma) + 1e-2      # we don't need this after soft clamp
        if temp != 1.:
            self.sigma *= temp

    def sample(self):
        return sample_normal_jit(self.mu, self.sigma)

    def sample_given_eps(self, eps):
        return eps * self.sigma + self.mu

    def log_p(self, samples, eps = 1e-5):
        normalized_samples = (samples - self.mu) / self.sigma
        log_p = - 0.5 * normalized_samples * normalized_samples - 0.5 * np.log(2 * np.pi) - torch.log(self.sigma)
        return log_p

    def kl(self, normal_dist):
        term1 = (self.mu - normal_dist.mu) / normal_dist.sigma
        term2 = self.sigma / normal_dist.sigma

        return 0.5 * (term1 * term1 + term2 * term2) - 0.5 - torch.log(term2)
    
    def to_device(self, device):
        self.mu = self.mu.to(device)
        self.sigma = self.sigma.to(device)
        return self

# Attention cell
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

'''
Input args:
    dim: length of input letents, e.g. 1024
    depth: number of layers
    heads: number of heads in multi-head attention
    dim_head: the inner length of each head in attention module, e.g. 64
    mlp_dim: the length of output latents, e.g. 1024
'''  
class AttnCell(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

#--------------------Main classes--------------------#
'''
VAE BACKBONE:
    input_channel: 1, input tensor is of shape (b, 1, L) = (batch_size, input_channel, input_length)
'''
class VAEBackbone(nn.Module):
    def __init__(self,
                 input_channel = 1, 
                 output_channel = 64,  
                 layers = [2,2,2,2], 
                 sampler_dim = 40, 
                 prior_scale = 1e-4, 
                 num_heads = 8,
                 dim_head = 64,
                 dropout = 0.25,
                 sample_decoder = False):
        super(VAEBackbone, self).__init__()
        
        num_layers = len(layers) # How many layers for encoder/decoder tower

        self.prior_scale = prior_scale # prior sclae for generating latents: in our experience, 1e-4 is a good choice.
        self.num_layers = num_layers
        self.sample_decoder = sample_decoder # should we sample the last layer of decoder? True: a learned std, False: a prior std of 1

        self.encoder_tower = nn.ModuleList()
        self.decoder_tower = nn.ModuleList()

        self.enc_combiners = nn.ModuleList()
        self.dec_combiners = nn.ModuleList()

        self.enc_sampler = nn.ModuleList()
        self.dec_sampler = nn.ModuleList()

        # get encoder tower
        '''
        Channel: 
            layer 1: 64 -> 64
            layer 2: 64 -> 128
            layer 3: 128 -> 256
            layer n > 3: 256 -> 256
        '''
        for n in range(num_layers):
            if n==0: # first layer of encoder, channels from 64 to 64
                Cin = output_channel
                Cout = output_channel
            else:
                Cin = output_channel*(np.power(2, min(n, 3)-1)) 
                Cout = output_channel*(np.power(2, min(n, 2)))
            self.encoder_tower.append(AttnCell(dim=Cin, depth = int(layers[n]), heads = num_heads, dim_head = dim_head, mlp_dim = Cout, dropout=dropout)) 
            self.enc_combiners.append(EncCombinerCell(Cout, Cout))
            self.enc_sampler.append(
                nn.Sequential(nn.Conv1d(Cout, sampler_dim, kernel_size=1, stride=1, padding=0, bias=True)))
        
        # get decoder tower
        '''
        Channel:
            layer n>3: 256 -> 256
            layer 3: 256 -> 128
            layer 2: 128 -> 64
            layer 1: 64 -> 64
        '''
        for n in range(num_layers):
            if n==num_layers-1: # last layer of decoder, channels from 64 to 64
                Cin = output_channel
                Cout = output_channel
            else:
                Cin = output_channel*(2**(min(num_layers-n-1, 2)))
                Cout = output_channel*(2**(min(num_layers-n-2, 2)))
            self.decoder_tower.append(AttnCell(dim=Cin, depth = int(layers[n]), heads = num_heads, dim_head = dim_head, mlp_dim = Cout, dropout=dropout))
            self.dec_combiners.append(DecCombinerCell(sampler_dim//2, Cin, Cin))
            if n<num_layers-1: # As the NVAE, the number of decoder equals the number of encoder minus 1
                self.dec_sampler.append(nn.Sequential(nn.Conv1d(Cout, sampler_dim, kernel_size=1, stride=1, padding=0, bias=True)))
        
        self.enc_0 = nn.Sequential(
            nn.ELU(),
            nn.Conv1d(in_channels=output_channel*(2**num_layers), out_channels=output_channel*(2**num_layers), kernel_size=1, stride=1, padding=0, bias=True),
            nn.ELU(),
        )

        # to generate prior z_1
        prior_ftr0_size = ((2**(min(num_layers-1, 2)))*output_channel, 1024) # should align with the siren latents' length
        self.prior_ftr0 = nn.Parameter(torch.rand(size=prior_ftr0_size), requires_grad=True)
        self.pre_processor = nn.Sequential(
            nn.Conv1d(in_channels=input_channel, out_channels=output_channel, kernel_size=1, stride=1, padding=0, bias=True),
        )
        self.post_processor = nn.Sequential(
            nn.Conv1d(in_channels=output_channel, out_channels=output_channel, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ELU(),
            nn.Conv1d(in_channels=output_channel, out_channels=input_channel, kernel_size=1, stride=1, padding=0, bias=True),
        )
        if sample_decoder:
            self.dist_embedding = ZCombiner(1024, 2048) # should align with the siren latents' length
    
    def forward(self, z_in, return_meta = False):
        # s1. we want to get posterior zn,...,z2,z1,z0 -> z0,z1,z2,...,zn
        z_n = self.pre_processor(z_in)
        z_posterior = []
        z_enc_combiners = []
        for n in range(self.num_layers):
            z_n = self.encoder_tower[n](z_n) # (b, 64->64->128->256, L)
            z_posterior.append(z_n)
            z_enc_combiners.append(self.enc_combiners[n])
        z_posterior.reverse()
        z_enc_combiners.reverse()

        # s2. get the first posterior
        param_0 = self.enc_0[0](z_posterior[0]) # q_0 (b, 256, 1024)
        param_0 = self.enc_sampler[self.num_layers-1](param_0) # q_0 (b, 40, 1024)
        mu_q, log_sigma_q = torch.chunk(param_0, 2, dim=1) # mu_q, log_sigma_q (b, 20, 1024)
        dist = Normal(mu_q, log_sigma_q)
        z,_ = dist.sample() # z (b, 20, 1024)
        log_q = dist.log_p(z) # (b, 20, 1024)
        all_q = [dist]
        all_log_q = [log_q]

        # s3. get the first prior
        dist = Normal(torch.zeros_like(mu_q), torch.ones_like(log_sigma_q)+torch.log(torch.tensor(self.prior_scale,device = mu_q.device)))
        log_p= dist.log_p(z) # (b, 20, 1024)
        all_p = [dist]
        all_log_p = [log_p]

        '''
        s is a random input sample, It doesn't affect the kld of p and q.
        For further exploration: s can be class label, hierarchy cluster.
        '''
        s = self.prior_ftr0.unsqueeze(0)
        batch_size = z.size(0)
        s = s.expand(batch_size, -1, -1)
        s = self.dec_combiners[0](s, z)
        s = self.decoder_tower[0](s)

        # s4. get the combined posterior from the features of generative model
        for n in range(1,self.num_layers):     
            # a1. get prior
            param = self.dec_sampler[n-1](s)
            mu_p, log_sigma_p = torch.chunk(param, 2, dim=1)

            # a2. get posterior
            ftr = z_enc_combiners[n](z_posterior[n], s)
            param = self.enc_sampler[self.num_layers-n-1](ftr)
            mu_q, log_sigma_q = torch.chunk(param, 2, dim=1)
            dist_q = Normal(mu_p + mu_q, log_sigma_p + log_sigma_q)
            # dist_q = Normal(mu_q, log_sigma_q)
            z, _ = dist_q.sample()
            log_q = dist_q.log_p(z)
            all_q.append(dist_q)
            all_log_q.append(log_q)

            dist_p = Normal(mu_p, log_sigma_p)
            log_p = dist_p.log_p(z)
            all_p.append(dist_p)
            all_log_p.append(log_p)

            # a3. combine posterior with prior features
            s = self.dec_combiners[n](s, z)
            s = self.decoder_tower[n](s)

        # the last combined s is sent to post_processor
        s = self.post_processor(s)

        # Sample decoder or not: True for a learned std, False for a prior std of 1
        if self.sample_decoder:
            z_mu, z_sigma = self.dist_embedding(s)
            out_dist = Normal(z_mu, z_sigma)
        else:
            out_dist = s

        kl_all = []
        kl_diag = []
        log_p, log_q = 0., 0.
        for q, p, log_q_conv, log_p_conv in zip(all_q, all_p, all_log_q, all_log_p):
            kl_per_var = q.kl(p)

            kl_diag.append(torch.mean(torch.sum(kl_per_var, dim=[1, 2]), dim=0))
            kl_all.append(torch.sum(kl_per_var, dim=[1, 2]))
            log_q += torch.sum(log_q_conv, dim=[1, 2])
            log_p += torch.sum(log_p_conv, dim=[1, 2])
        if return_meta:
            all_log_q = [torch.sum(log_q_conv, dim=[1, 2]) for log_q_conv in all_log_q]
            all_log_p = [torch.sum(log_p_conv, dim=[1, 2]) for log_p_conv in all_log_p]
            return out_dist, kl_all, all_q, all_p, all_log_q, all_log_p
        
        return out_dist, kl_all, kl_diag, log_q, log_p