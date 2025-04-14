# Transformer and Diffusion Transformer

### Vanilla Transformer
on enwik8
```
python train.py
```
Optionally, use Dynamic Tanh instead of RMSnorm or LayerNorm

Ref: [Transformer without Normalization](https://arxiv.org/abs/2503.10622)

Speed test: [Colab](https://colab.research.google.com/drive/1M_oksDjSleR0NDctWSPs5D6fQx6dkoaC?usp=sharing)

### Transformer with Infini-attention
```
python train_infini.py
```


### 2D Diffusion Transformer
on MNIST

RoPE for spatial embedding in 2D spatial attention
```
python train_dit.py
```

### 3D Diffusion Transformer
Sin-cos embedding before 3D attention
```
python train_dit_3d.py
```

### DiT variant
It has additional dim_head parameter in the DiT block.
Standard DiT:
```
# Head dimension is derived from hidden_size and num_heads
num_heads = hidden_size // dim_head
```
DiT variant:
```
# Head dimension is an independent parameter
# Can be set independently of hidden_size
```

Model Parameter Efficiency

Standard DiT:
- Required large hidden_size (e.g., 256) to maintain reasonable attention head dimensions
- Full model width applied to all operations (feed-forward, projections, etc.)

DiT variant:
- Can use much smaller hidden_size (e.g., 16) while maintaining attention capacity
- Significantly reduces parameter count in tokenization and feed-forward layers

### Flow Matching with Transformer
```
python train_flt.py
```

### Shortcut Model (Flow Matching) with Transformer
```
python train_shortcut.py
```

### Unet
```
python train_unet.py
```

## Result Analysis
[See doc](https://docs.google.com/document/d/1orGvXJ3iO-yDa6Szqt3DEdthMRKloXFkyRdi3fPn630/edit?usp=sharing)
