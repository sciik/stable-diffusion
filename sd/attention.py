import torch
from torch import nn
from torch.nn import funcional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, n_heads: int, d_model: int, in_proj_bias=True, out_proj_bias=True):
        super()._init__()
        
        self.in_proj = nn.Linear(d_model, 3*d_model, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=out_proj_bias)
        self.n_head = n_heads
        self.d_head = d_model // n_heads
    
    def forward(self, x: torch.Tensor, causal_mask=False):
        #x: (batch_size, seq_len, dim)
        batch_size, sequence_length, d_model = x.shape
        
        intermim_shape = (batch_size, sequence_length, self.n_head, self.d_head)
        
        # (batch_size, seq_len, dim) -> split this tensor (batch_size, seq_len, dim*3) -> (batch_size, seq_len, dim)
        Q, K, V = self.in_proj(x).chunk(3, dim=-1)
        
        # (batch_size, seq_len, dim) -> (batch_size, seq_len, n_head, d_head) -> (batch_size, seq_len, d_head, n_head)
        Q = Q.view(intermim_shape).transpose(1, 2)
        K = K.view(intermim_shape).transpose(1, 2)
        V = V.view(intermim_shape).transpose(1, 2)
        
        # (batch_size, n_head, seq_len, seq_len)
        weight = Q @ K.transpose(-1, -2)
        
        if causal_mask:
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)
        
        weight /= math.sqrt(self.d_head)
        weight = F.sotmax(weight, dim=-1)
        
        # (batch_size, n_head, seq_len, seq_len) @ (batch_size, n_head, seq_len, dim/h_head) -> (batch_size, n_head, seq_len, dim/n_head)
        output = weight @ V
        
        # (batch_size, n_head, seq_len, dim/n_head) -> (batch_size, seq_len, n_head, dim/n_head)
        output = output.transpose(1, 2)
        
        # (batch_size, seq_len, n_head, dim/n_head) -> (batch_size, seq_len, dim)
        output = output.reshape(x.shape)
        
        # (batch_size, seq_len, dim) -> (batch_size, seq_len, dim)
        output = self.out_proj(output)
        
        # (batch_size, seqlen, dim)
        return output


class CrossAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, d_cross: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__
        self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads
    
    def forward(self, x, y):
        # x(latent): (batch_size, seq_len_q, dim_q)
        # y(context): (batch_size, seq_len_kv, dim_kv) = (batch_size, 77, 768)
        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape
        
        interim_shape = (batch_size, -1, self.n_heads, self.d_head)
        
        #multiply by qw
        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)
        
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)
        
        w = q @ k.transpose(-1, -2)
        w /= math.sqrt(self.d_head)
        w = F.softmax(w, dim=-1)
        
        output = w @ v
        output = output.transpose(1, 2).contiguous()
        output = output.view(input_shape)
        output = self.out_proj(output)
        
        return output
