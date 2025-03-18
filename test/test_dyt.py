import torch
import torch.nn as nn
import time

# Define DynamicTanh as provided.
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


# A simple RMSNorm implementation for comparison.
class RMSNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-8):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.eps = eps

    def forward(self, x):
        # Compute the root mean square across the last dimension.
        norm = x.pow(2).mean(-1, keepdim=True).sqrt()
        x = x / (norm + self.eps)
        return x * self.weight

# Set parameters.
dim = 512           # Feature dimension.
batch_size = 32     # Number of samples.
seq_length = 128    # Sequence length.
num_trials = 1000   # Number of trials for timing.

# Determine device.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
# Initialize modules.
dynamic_tanh = DynamicTanh(dim).to(device)
rms_norm = RMSNorm(dim).to(device)

# Create random input.
input_tensor = torch.randn(batch_size, seq_length, dim).to(device)

# Warm-up iterations to ensure fair timing.
for _ in range(10):
    _ = dynamic_tanh(input_tensor)
    _ = rms_norm(input_tensor)

# Benchmarking function.
def benchmark(module, input_tensor, num_trials):
    times = []
    with torch.no_grad():
        for _ in range(num_trials):
            start = time.time()
            module(input_tensor)
            # Ensure all GPU operations have completed if using CUDA.
            if device == 'cuda':
                torch.cuda.synchronize()
            end = time.time()
            times.append(end - start)
    return sum(times) / len(times)

# Compute average forward pass times.
avg_time_rms = benchmark(rms_norm, input_tensor, num_trials)
avg_time_dynamic = benchmark(dynamic_tanh, input_tensor, num_trials)

print(f"Average time for DynamicTanh: {avg_time_dynamic * 1e6:.2f} microseconds")
print(f"Average time for RMSNorm: {avg_time_rms * 1e6:.2f} microseconds")