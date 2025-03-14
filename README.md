# Transformer and Diffusion Transformer

### Vanilla Transformer
on Wikitext
```
python train.py
```
Optionally, use Dynamic Tanh instead of RMSnorm or LayerNorm

Ref: [Transformer without Normalization](https://arxiv.org/abs/2503.10622)

### Transformer with Infini-attention
```
python train_infini.py
```


### 2D Diffusion Transformer
on MNIST

RoPE for spatial embedding
```
python train_dit.py
```

### 3D Diffusion Transformer
```
python train_dit_3d.py
```
