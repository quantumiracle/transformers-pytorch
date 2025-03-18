from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def flow_matching_schedules(T):
    """
    Returns pre-computed schedules for flow matching sampling and training process.
    """
    # Linear time steps from 0 to 1
    t = torch.linspace(0, 1, T + 1)
    
    return {
        "t": t,  # time steps from 0 to 1
    }

def extract(a, t, x_shape):
    f, b = t.shape
    out = a[t]
    return out.reshape(f, b, *((1,) * (len(x_shape) - 2)))

class FlowMatching(nn.Module):
    def __init__(self, model, n_T, device, drop_prob=0.1, add_velocity_direction_loss=False, 
                 lognorm_t=False, target_std=1.0):
        super(FlowMatching, self).__init__()
        self.model = model.to(device)

        # register_buffer allows accessing dictionary produced by flow_matching_schedules
        for k, v in flow_matching_schedules(n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()
        self.add_velocity_direction_loss = add_velocity_direction_loss
        self.lognorm_t = lognorm_t
        self.target_std = target_std

    def lognorm_sample(self, size):
        """
        Sample from logit-normal distribution
        """
        # Sample from normal distribution
        samples = torch.randn(size).to(self.device)
        
        # Transform to 0 to 1 using sigmoid
        samples = 1.0 / (1.0 + torch.exp(-samples))
        return samples

    def get_flow_field(self, x0, x1, t):
        """
        Compute the straight-line flow field between x0 and x1 at time t
        """
        return x1 - x0
    
    def get_xt(self, x0, x1, t):
        """
        Compute the intermediate point at time t between x0 and x1
        """
        return x0 + t * (x1 - x0)

    def forward(self, x0, x1, c):
        """
        This method is used in training, samples t randomly and computes loss
        """
        # Apply data shifting: scale data to target standard deviation
        if self.target_std != 1.0:
            # Calculate current standard deviation
            # https://github.com/hustvl/LightningDiT/blob/959d2ca76f238023bc9ff17ff2dcd094fe62be9b/vavae/ldm/models/diffusion/ddpm.py#L489
            data_std = x1.flatten().std()
            # Scale data to target standard deviation
            x1 = x1 * (self.target_std / (data_std + 1e-6))
        
        if x0.ndim == 4:  # (B, C, H, W)
            if self.lognorm_t:
                _ts = self.lognorm_sample((x0.shape[0], 1, 1, 1))
            else:
                _ts = torch.rand(x0.shape[0], 1, 1, 1).to(self.device)  # t ~ Uniform(0, 1)
        elif x0.ndim == 5:  # (B, T, C, H, W)
            if self.lognorm_t:
                _ts = self.lognorm_sample((x0.shape[0], x0.shape[1], 1, 1, 1))
            else:
                _ts = torch.rand(x0.shape[0], x0.shape[1], 1, 1, 1).to(self.device)  # t ~ Uniform(0, 1)
        else:
            raise ValueError(f"x.ndim must be 4 or 5, but got {x0.ndim}")
        
        # Compute x_t at the sampled time steps
        x_t = self.get_xt(x0, x1, _ts)
        
        # Compute the target vector field (straight line from x0 to x1)
        target_field = self.get_flow_field(x0, x1, _ts)
        
        # Dropout context with some probability
        context_mask = torch.bernoulli(torch.ones_like(c) - self.drop_prob).to(self.device)
        
        # Get model prediction
        pred_field = self.model(x_t, _ts.squeeze(), c, context_mask)
        
        # MSE loss
        mse_loss = self.loss_mse(target_field, pred_field)
        
        # Add velocity direction loss if enabled
        if self.add_velocity_direction_loss:
            # Compute cosine similarity between predicted and target fields
            # Flatten spatial dimensions for cosine similarity
            target_flat = target_field.view(target_field.shape[0], -1)
            pred_flat = pred_field.view(pred_field.shape[0], -1)
            
            # Normalize vectors for cosine similarity
            target_norm = torch.nn.functional.normalize(target_flat, dim=1)
            pred_norm = torch.nn.functional.normalize(pred_flat, dim=1)
            
            # Compute cosine similarity
            cos_sim = (target_norm * pred_norm).sum(dim=1)
            
            # Direction loss: 1 - cosine similarity
            direction_loss = 1.0 - cos_sim.mean()
            
            # Combined loss
            return mse_loss + direction_loss
        else:
            return mse_loss

    def sample(self, n_sample, size, device, guide_w=0.0, cond=None, steps=None):
        """
        Sample from the flow model using Euler integration
        """
        if steps is None:
            steps = self.n_T
            
        # Start with random noise
        x_i = torch.randn(n_sample, *size).to(device)
        n_frames = x_i.shape[1] if len(size) > 3 else 1
        
        # Don't drop context at test time
        context_mask = torch.ones_like(cond).to(device)
        
        # Double the batch for classifier-free guidance
        cond = cond.repeat(2, 1)  # (2B, cond_dim)
        context_mask = context_mask.repeat(2, 1)  # (2B, 1)
        context_mask[n_sample:] = 0.  # Makes second half of batch context free
        
        # Time step size for Euler integration
        dt = 1.0 / steps
        
        x_i_store = []  # Keep track of generated steps in case want to plot something
        
        # Euler integration
        for i in range(steps):
            t_i = i * dt
            print(f'sampling timestep {i+1}/{steps}, t={t_i:.4f}', end='\r')
            
            # Current time step tensor
            t_is = torch.ones(n_sample) * t_i
            t_is = t_is.to(device)
            
            # Double batch
            if x_i.ndim == 4:
                x_i = x_i.repeat(2, 1, 1, 1)
                t_is = t_is.repeat(2)
            elif x_i.ndim == 5:
                x_i = x_i.repeat(2, 1, 1, 1, 1)
                t_is = t_is.repeat(2).unsqueeze(1).repeat(1, n_frames)
            else:
                raise ValueError(f"x_i.ndim must be 4 or 5, but got {x_i.ndim}")
            
            # Get vector field prediction
            v = self.model(x_i, t_is, cond, context_mask)
            
            # Split predictions and compute weighting for classifier-free guidance
            v1 = v[:n_sample]
            v2 = v[n_sample:]
            v = (1 + guide_w) * v1 - guide_w * v2
            
            # Keep only the first half of the batch
            x_i = x_i[:n_sample]
            
            # Euler step
            x_i = x_i + v * dt
            
            # Store intermediate results
            if i % 20 == 0 or i == steps - 1 or i < 8:
                x_i_store.append(x_i.detach().cpu().numpy())
        
        x_i_store = np.array(x_i_store)
        return x_i, x_i_store
