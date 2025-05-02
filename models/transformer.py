from __future__ import annotations
from typing import Callable

from math import ceil
from copy import deepcopy
from functools import partial
from collections import namedtuple

import tqdm

import torch
from torch import nn, stack, cat
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Linear
from einops import repeat, rearrange, pack, unpack
from einops.layers.torch import Rearrange
from rotary_embedding_torch import RotaryEmbedding

LinearNoBias = partial(Linear, bias = False)

# helpers

def exists(v):
    return v is not None

# min_p
# https://arxiv.org/abs/2407.01082

def min_p_filter(logits, min_p = 0.1):
    probs = logits.softmax(dim = -1)
    max_probs = probs.amax(dim = -1, keepdim = True)
    limit = min_p * max_probs
    return torch.where(probs < limit, float('-inf'), logits)

# sampling related

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.rand_like(t)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1.):
    if temperature > 0.:
        t = t / temperature + gumbel_noise(t)
    return t.argmax(dim = -1, keepdim = True)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class GEGLU(Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return F.silu(gate) * x

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, is_causal = True):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.rotary_emb = RotaryEmbedding(dim_head)
        self.attend = nn.Softmax(dim = -1)

        self.is_causal = is_causal

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x, cache = None):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        # caching
        if exists(cache):
            past_k, past_v = cache
            k = torch.cat((past_k, k), dim = -2)
            v = torch.cat((past_v, v), dim = -2)

        new_cache = (k, v)

        # relative positions
        q, k = self.rotary_emb.rotate_queries_with_cached_keys(q, k)

        # causal mask
        if self.is_causal:
            seq_len = q.shape[-2]
            causal_mask = torch.triu(torch.ones((seq_len, seq_len), device=x.device), diagonal=1).bool()

        # attention
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if self.is_causal:
            dots.masked_fill_(causal_mask, float('-inf'))

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        return self.to_out(out), new_cache


class DynamicTanh(nn.Module):
    def __init__(self, normalized_shape, channels_last=True, alpha_init_value=0.5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value
        self.channels_last = channels_last

        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        if self.channels_last:
            x = x * self.weight + self.bias
        else:
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
        return x

    def extra_repr(self):
        return f"normalized_shape={self.normalized_shape}, alpha_init_value={self.alpha_init_value}, channels_last={self.channels_last}"


class Transformer(Module):
    def __init__(
        self, 
        num_tokens,
        dim, 
        depth, 
        heads, 
        dim_head, 
        mlp_dim,
        dynamic_tanh = False,
        token_emb: Module | None = None,
        ):
        super().__init__()
        if dynamic_tanh:
            self.norm = DynamicTanh(dim)
        else:
            self.norm = nn.RMSNorm(dim) # or LayerNorm
        if not exists(token_emb):
            token_emb = nn.Embedding(num_tokens, dim)

        self.token_emb = token_emb

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))

        self.to_logits = LinearNoBias(dim, num_tokens)

    @torch.no_grad()
    def sample(
        self,
        prompt: Tensor,
        seq_len: int,
        temperature = 1.5,
        filter_fn: Callable = min_p_filter,
        filter_kwargs: dict = dict(
            min_p = 0.1,
        ),
        use_cache = False,
        show_progress = True,
        ):
        was_training = self.training
        self.eval()

        cache = None
        assert len(prompt.shape) == 2

        prompt_seq_len, out = prompt.shape[-1], prompt.clone()
        sample_num_times = max(0, seq_len - prompt_seq_len)

        with tqdm.tqdm(total = sample_num_times, disable = not show_progress) as pbar:
            while out.shape[-1] < seq_len:
                logits, next_cache = self.forward(
                    out,
                    cache = cache,
                    return_loss = False,
                    return_cache = True,
                )

                if use_cache:
                    cache = next_cache

                if not exists(logits):
                    continue

                logits = logits[:, -1]

                logits = filter_fn(logits, **filter_kwargs)
                sample = gumbel_sample(logits, temperature = temperature)
                out = torch.cat((out, sample), dim = -1)
                pbar.update(1)

        self.train(was_training)

        return out[..., prompt_seq_len:]

    def forward(
        self, 
        x,
        cache = None,
        return_loss = False,
        return_cache = False
        ):
        if return_loss:
            x, labels = x[:, :-1], x[:, 1:]

        x = self.token_emb(x)

        is_inferencing = exists(cache)

        # when inferencing with cache, only do one token at a time
        if is_inferencing:
            x = x[:, -1:]
        
        if not exists(cache):
            cache = [None] * len(self.layers)
        next_cache = []

        for i, (attn, ff) in enumerate(self.layers):
            pre_cache = cache[i] if exists(cache) else None
            attn_out, cache_per_layer = attn(x, cache = pre_cache)
            x = attn_out + x
            x = ff(x) + x
            next_cache.append(cache_per_layer)

        x = self.norm(x)
        logits = self.to_logits(x)

        next_cache = next_cache if return_cache else None

        if not return_loss:
            if not return_cache:
                return logits

            return logits, next_cache

        return F.cross_entropy(rearrange(logits, 'b n l -> b l n'), labels)
