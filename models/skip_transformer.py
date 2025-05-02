# skip_transformer.py
#
# A “next‑N‑token” hybrid Transformer that:
#  • reads an input sequence of length L=l-n:  x₁…x_{l‑N}
#  • outputs sequence of length L=l-n:  x_{N+1}…x_l   (length N + overlap)
#  • keeps causal masking only where needed (overlap zone)
#  • lets the N brand‑new tokens attend to each other freely
#
# Segment layout if   l = total length   and   N = skip size, attention matrix  LxL, L=l-n
#
#   ┌───────────────┬─────────────┬────────────┐
#   │ early context │  overlap    │  new block │
#   │   length N    │ l–2N tokens │  length N  │
#   └───────────────┴─────────────┴────────────┘
# Example: l = 10, N = 3, L=l-n=7
# input sequence:  x1, x2, x3, x4, x5, x6, x7
# output sequence: x4, x5, x6, x7, x8, x9, x10
# attention mask:
#    x1, x2, x3, x4, x5, x6, x7
# x4, 0,  0,  0,  0,  1,  1,  1
# x5, 0,  0,  0,  0,  0,  1,  1
# x6, 0,  0,  0,  0,  0,  0,  1
# x7, 0,  0,  0,  0,  0,  0,  0
# x8, 0,  0,  0,  0,  0,  0,  0
# x9, 0,  0,  0,  0,  0,  0,  0
# x10,0,  0,  0,  0,  0,  0,  0
# Requirements:
#   einops, torch
#   an existing `Transformer` core with (attn, ff) layers that accept an
#   explicit Boolean attention mask of shape (seq, seq)
#
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import Callable, List, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn
from torch.nn import Module, ModuleList, Linear
import tqdm
from models.transformer import Transformer, exists, gumbel_sample, min_p_filter, DynamicTanh, RotaryEmbedding, FeedForward, LinearNoBias


# ---------------------------------------------------------------------
# helper: build the hybrid mask from only N and the full length L
# ---------------------------------------------------------------------
def build_hybrid_mask(N: int, L: int, *, device=None) -> torch.BoolTensor:
    """
    Hybrid causal mask for next‑N‑token prediction.

    Args:
        N (int) : skip size  (also early‑context length and new‑token length)
        L (int) : sequence length
        device : torch device

    Returns:
        Bool tensor of shape (L, L) where True = masked‑out, False = attend
    """
    mask = torch.zeros((L, L), dtype=torch.bool, device=device)

    # causal mask
    for i in range(L-N):
        mask[i, N+i:] = True

    return mask
# ---------------------------------------------------------------------

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.rotary_emb = RotaryEmbedding(dim_head)
        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x, cache = None, mask = None):
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

        # attention
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if exists(mask):
            dots.masked_fill_(mask, float('-inf'))

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        return self.to_out(out), new_cache

class SkipTransformer(Module):
    def __init__(
        self, 
        num_tokens,
        N, # skip size
        dim, 
        depth, 
        heads, 
        dim_head, 
        mlp_dim,
        dynamic_tanh = False,
        token_emb: Module | None = None,
        ):
        super().__init__()
        self.N = N
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

    # ------------------------------------------------------------
    # forward pass
    # ------------------------------------------------------------

    def forward(
        self, 
        x,
        cache = None,
        return_loss = False,
        return_cache = False
        ):
        if return_loss:
            x, labels = x[:, :-self.N], x[:, self.N:]

        x = self.token_emb(x)

        # build mask once
        hybrid_mask = build_hybrid_mask(self.N, x.size(1), device=x.device)

        is_inferencing = exists(cache)

        # when inferencing with cache, only do one token at a time
        if is_inferencing:
            x = x[:, -self.N:]
        
        if not exists(cache):
            cache = [None] * len(self.layers)
        next_cache = []

        for i, (attn, ff) in enumerate(self.layers):
            pre_cache = cache[i] if exists(cache) else None
            attn_out, cache_per_layer = attn(x, cache = pre_cache, mask = hybrid_mask)
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


    # ------------------------------------------------------------
    # sampling N tokens at a time
    # ------------------------------------------------------------

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

                logits = logits[:, -self.N:]

                new_tokens = []
                for i in range(self.N):
                    sample = filter_fn(logits[:, i], **filter_kwargs)
                    sample = gumbel_sample(sample, temperature = temperature)
                    new_tokens.append(sample)
                    # out = torch.cat((out, sample), dim = -1)
                new_tokens = torch.cat(new_tokens, dim = 1)
                out = torch.cat((out, new_tokens), dim = 1)
                pbar.update(1)

        self.train(was_training)

        return out[..., prompt_seq_len:]
    