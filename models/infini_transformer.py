from __future__ import annotations
from typing import Callable

from math import ceil
from copy import deepcopy
from functools import partial
from collections import namedtuple
from torch import einsum
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



def round_up_multiple(seq, mult):
    return ceil(seq / mult) * mult

def pad_and_segment_with_inverse(
    seq,
    segment_len,
    fold_into_batch = True,
    inverse_remove_pad = True
):
    batch, seq_len = seq.shape[:2]
    next_seq_len_mult = round_up_multiple(seq_len, segment_len)

    padding = next_seq_len_mult - seq_len
    needs_pad = padding > 0

    if needs_pad:
        seq = F.pad(seq, (0, 0, 0, padding))

    if fold_into_batch:
        seq = rearrange(seq, 'b (w n) d -> (b w) n d', n = segment_len)

    def inverse(out):

        if fold_into_batch:
            out = rearrange(out, '(b w) ... n d -> b ... (w n) d', b = batch)

        if needs_pad and inverse_remove_pad:
            out = out[..., :-padding, :]

        return out

    return seq, inverse

def sigma_act(x):
    return F.elu(x) + 1

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

def updateMemory(kv_mem, k, v, z, delta=False):
    if kv_mem is not None:
        assert kv_mem.shape[1] == v.shape[1], "Memory length must be the same as the number of new keys"
    if delta and (kv_mem is not None):
        sigma_k = sigma_act(k)
        numerator = einsum(
            "b h n k, b h k v -> b h n v",
            sigma_k,
            kv_mem,
        )
        denominator = einsum(
            "b h n k, b h k -> b h n",
            sigma_k,
            z,
        )
        denominator = rearrange(
            denominator,
            "b h n -> b h n 1",
        )
        prev_v = numerator / denominator
        new_value_states = v - prev_v
        new_kv_mem = torch.matmul(sigma_k.transpose(-2, -1), new_value_states)
    else:
        k_T = rearrange(sigma_act(k), "b h n d -> b h d n")
        new_kv_mem = sigma_act(k_T) @ v
    if kv_mem is not None:
        new_kv_mem = kv_mem + new_kv_mem
    return new_kv_mem

def getAttnMem(kv_mem, z, q):
    assert kv_mem is not None, "Attention memory must be provided"
    sigma_q = sigma_act(q)
    retrieved_memory = einsum(
        "b h n k, b h k v -> b h n v",
        sigma_q,
        kv_mem,
    )
    denominator = einsum(
        "b h n k, b h k -> b h n",
        sigma_q,
        z,
    )
    denominator = rearrange(
        denominator,
        "b h n -> b h n 1",
    )
    retrieved_memory = retrieved_memory / (denominator + 1e-6)  # Adding 1e-6 for preventing division to 0
    return retrieved_memory


def updateZ(z, k):
    # k: (b, h, n, d)
    k = sigma_act(k)
    new_z = torch.sum(k, dim=2, keepdim=False)
    if z is not None:
        new_z = z + new_z
    return new_z #

class InfiniAttention(nn.Module):
    """
    InfiniAttention is a variant of the Attention class that allows for fixed length of the cached keys and values as memory.
    https://arxiv.org/pdf/2404.07143
    """
    def __init__(
        self, 
        dim, 
        heads = 8, 
        dim_head = 64, 
        is_causal = True, 
        bptt = True, 
        delta_update = True,
        segment_len = 32,
        memory_pe = False,
    ):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.rotary_emb = RotaryEmbedding(dim_head)
        self.attend = nn.Softmax(dim = -1)
        self.dim_head = dim_head
        self.memory_pe = memory_pe
        self.is_causal = is_causal
        self.bptt = bptt
        self.delta_update = delta_update
        self.segment_len = segment_len
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

        self.device = torch.device("cuda")

        # memory params
        self.mem_beta = nn.Parameter(torch.zeros(1, heads, 1, 1)).to(self.device)

    def forward(self, x, cache = None, memory_cache = None):
        batch = x.shape[0]

        # auto pad to multiple
        x, inverse_segment = pad_and_segment_with_inverse(x, self.segment_len, fold_into_batch = False)

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        # caching
        if exists(cache):
            past_k, past_v = cache
            k = torch.cat((past_k, k), dim = -2)
            v = torch.cat((past_v, v), dim = -2)
        next_cache = tuple(map(inverse_segment, (k, v)))

        # relative positions
        if not self.memory_pe:
            q_no_pe, k_no_pe = tuple(rearrange(t, 'b h (w n) d -> w b h n d', n = self.segment_len) for t in (q, k))

        q, k = self.rotary_emb.rotate_queries_with_cached_keys(q, k)

        # segment
        q, k, v = tuple(rearrange(t, 'b h (w n) d -> w b h n d', n = self.segment_len) for t in (q, k, v))

        # initialize memory and z
        if memory_cache is None:
            mem = torch.zeros(batch, self.heads, self.dim_head, self.dim_head).to(self.device)
            z = torch.zeros(batch, self.heads, self.dim_head).to(self.device) + 1e-6 # as denominator, avoid division to 0
        else:
            mem, z = memory_cache

        outputs = []
        # it's a loop version of infinite attention, cannot parallelize due to memory sequence nature
        # https://github.com/vmarinowski/infini-attention/blob/main/infini_attention/infini_attention.py
        # other code may parallel with segment: https://github.com/Beomi/InfiniTransformer/blob/045f4eba17a3155dbdc255d76b4623bebc768eb6/infini_llama/modeling_infini_llama.py#L975
        for i in range(k.shape[0]):  # number of segments
            q_slice = q[i]
            k_slice = k[i]
            v_slice = v[i]

            # a simpler variant for attention computation
            out = F.scaled_dot_product_attention(query=q_slice, key=k_slice, value=v_slice, is_causal=self.is_causal)

            if not self.memory_pe: # no pe for memory retrieval and update
                q_slice = q_no_pe[i]
                k_slice = k_no_pe[i]

            # memory retrieval (if update before retrieval, the information in segment is leaked and no longer causal; low training loss but high inference error.)
            mem_out = getAttnMem(mem, z, q_slice)
            out = F.sigmoid(self.mem_beta) * mem_out + (1 - F.sigmoid(self.mem_beta)) * out
            outputs.append(out)

            # memory update
            if self.bptt:
                mem = updateMemory(mem, k_slice, v_slice, z, self.delta_update)
                z = updateZ(z, k_slice)
            else:
                with torch.no_grad():
                    mem = updateMemory(mem, k_slice, v_slice, z, self.delta_update)
                    z = updateZ(z, k_slice)

        out = torch.stack(outputs)
        out = rearrange(out, 'w b h n d -> (w b) n (h d)', b = batch)

        out = self.to_out(out)
        out = rearrange(out, '(w b) n d -> b (w n) d', b = batch)

        out = inverse_segment(out)

        return out, next_cache

class InfiniTransformer(Module):
    def __init__(
        self, 
        num_tokens,
        dim, 
        depth, 
        heads, 
        dim_head, 
        mlp_dim,
        segment_len,
        token_emb: Module | None = None,
        memory_pe = False,
        bptt = True,
        delta_update = True,
        ):
        super().__init__()
        self.norm = nn.RMSNorm(dim) # or LayerNorm
        if not exists(token_emb):
            token_emb = nn.Embedding(num_tokens, dim)

        self.token_emb = token_emb

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                InfiniAttention(dim, heads = heads, dim_head = dim_head, segment_len = segment_len, memory_pe = memory_pe, bptt = bptt, delta_update = delta_update),
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
                logits, next_cache = self.forward(  # TODO: add memory cache
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
