import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm, trange
import copy
from .Functions import *
from .evaluation import *
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from .Modules import *
from flash_attn import flash_attn_func
from .rotary import RotaryEmbedding


class SigmoidScale(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        scale = x.shape[-1]
        return torch.sigmoid(x) / scale

class RelMultiHeadAttention(nn.Module):
    ''' Relative Multi-Head Attention module '''

    def __init__(
            self,
            input_dim,
            n_head,
            d_model,
            d_k,
            d_v,
            dropout,
            simple_attention=False):
        super().__init__()

        self.scale = d_k ** -0.5
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(input_dim, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(input_dim, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(input_dim, n_head * d_v, bias=False)

        nn.init.normal_(self.w_qs.weight, mean=0,
                        std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0,
                        std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0,
                        std=np.sqrt(2.0 / (d_model + d_v)))

        self.w_out = nn.Linear(n_head * d_v, d_model)
        nn.init.zeros_(self.w_out.weight)
        nn.init.zeros_(self.w_out.bias)

        self.attn_dropout = dropout
        self.rotary_emb = RotaryEmbedding(dim = d_k, seq_before_head_dim=True)

        self.simple_attention = simple_attention
        if self.simple_attention:
            self.softmax = nn.Softmax(dim=-1)
            self.shap_register=True

    def forward(self, q, k=None, v=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        if k is None:
            k = q
        if v is None:
            v = q

        sz_b, len_q, _ = q.shape
        sz_b, len_k, _ = k.shape
        sz_b, len_v, _ = v.shape

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k) * self.scale
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = self.rotary_emb.rotate_queries_or_keys(q)
        k = self.rotary_emb.rotate_queries_or_keys(k)

        if not self.simple_attention:
            # out: (batch_size, seqlen, nheads, headdim).
            out = self.w_out(flash_attn_func(q.half(),
                                             k.half(),
                                             v.half(),
                                             dropout_p=self.attn_dropout).float().contiguous().view(sz_b, len_q, -1))
        else:
            attn = torch.einsum('bnqd,bnkd->bnqk', q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3))
            attn = self.softmax(attn)
            attn = self.attn_dropout(attn)
            out = torch.einsum('bnqk, bnkd->bnqd', attn, v.permute(0, 2, 1, 3))
            out = self.w_out(out.permute(0, 2, 1, 3).contiguous().view(sz_b, len_q, -1))



        return out


'''
This is a custom implementation of element-wise multiplication for shap
'''
class ElementWiseMultiply(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, attn):
        return x * attn

class AttentionPool(nn.Module):
    def __init__(self, dim, pool_size = 2):
        super().__init__()
        self.pool_size = pool_size
        self.pool_fn = Rearrange('b d (n p) -> b d n p', p = pool_size)

        self.to_attn_logits = nn.Conv2d(dim, dim, 1, bias=False)
        self.dim = dim
        nn.init.dirac_(self.to_attn_logits.weight)

        with torch.no_grad():
            self.to_attn_logits.weight.mul_(2)

        self.softmax = nn.Softmax(dim=-1)
        self.element_wise_multiply = ElementWiseMultiply()

    def forward(self, x):
        b, _, n = x.shape
        remainder = n % self.pool_size
        needs_padding = remainder > 0
        if needs_padding:
            x = F.pad(x, (0, remainder), value = 0)
            mask = torch.zeros((b, 1, n), dtype = torch.bool, device = x.device)
            mask = F.pad(mask, (0, remainder), value = True)

        x = self.pool_fn(x)
        # logits = F.conv2d(x, self.to_attn_logits)
        logits = self.to_attn_logits(x)
        if needs_padding:
            mask_value = -torch.finfo(logits.dtype).max
            logits = logits.masked_fill(self.pool_fn(mask), mask_value)

        attn = self.softmax(logits)
        out = self.element_wise_multiply(x, attn).sum(dim = -1)
        return out


def ConvBlock(dim,
              dim_out = None,
              kernel_size = 1,
              batch_norm = False,
              ):
    if dim_out is None:
        dim_out = dim
    return nn.Sequential(
        nn.BatchNorm1d(dim) if batch_norm else nn.Identity(),
        nn.GELU(),
        nn.Conv1d(dim, dim_out, kernel_size, padding = kernel_size // 2)
    )


class DNA_CNN_Enformer(nn.Module):
    """
    This is actually as simple as one CNN layer,
    It's used to extract the DNA sequence features (the first layer)
    just to keep the consistency using the Module way of construction
    """
    def __init__(self,
                 n_filters=64,
                 kernel_size=21,
                 padding=10,
                 in_channels=4,
                 pool_size=2,
                 batch_norm=False,
                 ):

        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels,
                      n_filters,
                      kernel_size, padding=padding),
            Residual(ConvBlock(n_filters, batch_norm=batch_norm)),
            AttentionPool(n_filters, pool_size=pool_size) if pool_size > 1 else nn.Identity(),
        )
        self.activation = nn.GELU()


    def forward(self, X):
        X = self.stem(X)
        X = self.activation(X)
        return X


class ConvTower(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 pool_size,
                 n_layers,
                 batch_norm=False,):
        super().__init__()
        base = math.exp(math.log(out_channels / in_channels) / (n_layers - 1))
        kernel_nums = [in_channels] + [int(in_channels * base ** i) for i in
                                                     range(n_layers)]
        kernel_nums[-1] = out_channels
        conv_layers = []
        for dim_in, dim_out in zip(kernel_nums[:-1], kernel_nums[1:]):
            conv_layers.append(nn.Sequential(
                ConvBlock(dim_in, dim_out, kernel_size=kernel_size, batch_norm=batch_norm),
                Residual(ConvBlock(dim_out, dim_out, 1, batch_norm=batch_norm)),
                AttentionPool(dim_out, pool_size=pool_size) if pool_size > 1 else nn.Identity(),
            ))

        self.conv_tower = nn.Sequential(*conv_layers)

    def forward(self, X):
        return self.conv_tower(X)


class AttentionTower(nn.Module):
    def __init__(self,
                 input_dim,
                 n_head,
                 d_model,
                 d_k,
                 attn_dropout,
                 pos_dropout,
                 dropout,
                 n_layers,
                 simple_attention=False,):
        super().__init__()
        transformer = []
        attention = RelMultiHeadAttention
        for _ in range(n_layers):
            transformer.append(nn.Sequential(
                Residual(nn.Sequential(
                    nn.LayerNorm(d_model),
                    attention(
                        input_dim=input_dim,
                        n_head=n_head,
                        d_model=d_model,
                        d_k=d_k,
                        d_v=d_k,
                        dropout=attn_dropout,
                    ),
                    nn.Dropout(dropout)
                )),
                Residual(nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, d_model * 2),
                    nn.Dropout(dropout),
                    nn.ReLU(),
                    nn.Linear(d_model * 2, d_model),
                    nn.Dropout(dropout)
                ))
            ))

        self.transformer = nn.Sequential(*transformer)

    def forward(self, X):
        return self.transformer(X)
