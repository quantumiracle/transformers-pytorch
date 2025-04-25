import argparse
import random
import torch
import gzip
import numpy as np
import os
from models.neuromem_transformer import NeuromemTransformer

# Command line arguments
parser = argparse.ArgumentParser(description='Inference with NeuromemTransformer')
parser.add_argument('--model_path', type=str, required=True, help='Path to the saved model')
parser.add_argument('--prompt', type=str, default='', help='Text prompt to start generation')
parser.add_argument('--prompt_length', type=int, default=100, help='Length of random prompt if no prompt is provided')
parser.add_argument('--generation_length', type=int, default=512, help='Length of text to generate')
parser.add_argument('--temperature', type=float, default=1.5, help='Temperature for sampling')
parser.add_argument('--min_p', type=float, default=0.1, help='Min-p filtering threshold')
parser.add_argument('--gpu', type=int, default=-1, help='GPU ID to use (-1 for CPU)')
parser.add_argument('--dynamic_tanh', action='store_true', help='Use dynamic tanh activation')
parser.add_argument('--data_path', type=str, default='./data/enwik8.gz', help='Path to enwik8.gz for random prompts')
args = parser.parse_args()

# Device configuration
if args.gpu >= 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cpu')

# Helper functions
def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))

# First check if the model was saved with dynamic_tanh
# Load the state dict to inspect its keys
state_dict = torch.load(args.model_path, map_location=device)
has_dynamic_tanh = 'norm.alpha' in state_dict

if has_dynamic_tanh and not args.dynamic_tanh:
    print("Detected DynamicTanh in saved model. Setting --dynamic_tanh=True.")
    args.dynamic_tanh = True
elif not has_dynamic_tanh and args.dynamic_tanh:
    print("Warning: Model was not saved with DynamicTanh, but --dynamic_tanh=True was specified.")
    print("Setting --dynamic_tanh=False to match the saved model.")
    args.dynamic_tanh = False

# Model initialization and loading
model = NeuromemTransformer(
    num_tokens=256,
    dim=384,
    depth=8,
    heads=8,
    dim_head=64,
    mlp_dim=512,
    dynamic_tanh=args.dynamic_tanh,
).to(device)

# Load model weights
model.load_state_dict(state_dict)
model.eval()

# Calculate perplexity from loss
def calculate_perplexity(loss):
    return torch.exp(loss).item()

def compute_perplexity(tokens):
    """Compute perplexity on a given token sequence"""
    with torch.no_grad():
        # Convert to tensor if needed
        if not isinstance(tokens, torch.Tensor):
            tokens = torch.tensor(tokens, dtype=torch.long)
        
        # Ensure tensors are properly shaped
        if len(tokens.shape) == 1:
            tokens = tokens.unsqueeze(0)
        
        tokens = tokens.to(device)
        
        # NeuromemTransformer handles the loss calculation internally
        # It automatically creates the shifted targets from the input
        loss = model(tokens, return_loss=True)
        
        # Calculate perplexity from loss
        perplexity = calculate_perplexity(loss)
        
        return loss.item(), perplexity

def get_random_prompt():
    """Load a random prompt from the validation dataset and return the prompt tensor and full data"""
    with gzip.open(args.data_path) as file:
        data = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()
        _, data_val = np.split(data, [int(90e6)])
        data_val = torch.from_numpy(data_val)
    
    rand_start = torch.randint(0, data_val.size(0) - (args.prompt_length + args.generation_length), (1,))
    prompt_tensor = data_val[rand_start: rand_start + args.prompt_length].long()
    
    # Also get the ground truth continuation
    ground_truth = data_val[rand_start + args.prompt_length: rand_start + args.prompt_length + args.generation_length].long()
    
    return prompt_tensor, ground_truth, rand_start

def generate_text():
    """Generate text based on a prompt"""
    ground_truth = None
    rand_start = None
    
    # Use provided prompt or get a random one from validation data
    if args.prompt:
        # Convert text to tensor
        prompt_tensor = torch.tensor([ord(c) for c in args.prompt], dtype=torch.long).unsqueeze(0).to(device)
        prompt_text = args.prompt
    else:
        # Get random prompt with ground truth
        prompt_tensor, ground_truth, rand_start = get_random_prompt()
        prompt_tensor = prompt_tensor.unsqueeze(0).to(device)
        prompt_text = decode_tokens(prompt_tensor[0].tolist())
    
    print(f"\nPrompt:\n{prompt_text}")
    print("-" * 80)
    
    # Generate text
    with torch.no_grad():
        sample = model.sample(
            prompt_tensor,
            args.generation_length,
            temperature=args.temperature,
            filter_kwargs=dict(min_p=args.min_p),
            show_progress=True
        )
    
    generated_text = decode_tokens(sample[0].tolist())
    
    # Calculate perplexity on ground truth if available
    if ground_truth is not None and len(ground_truth) > 1:
        loss, perplexity = compute_perplexity(ground_truth)
        print(f"\nGround Truth Perplexity: {perplexity:.2f} (Loss: {loss:.4f})")
    
    if ground_truth is not None:
        ground_truth_text = decode_tokens(ground_truth.tolist())
    else:
        ground_truth_text = None
        
    return prompt_text, generated_text, ground_truth_text

if __name__ == "__main__":
    print(f"Using device: {device}")
    print(f"Model loaded from: {args.model_path}")
    print(f"Dynamic tanh: {args.dynamic_tanh}")
    
    prompt_text, generated_text, ground_truth_text = generate_text()
    
    print("\nGenerated Text:")
    print("-" * 80)
    print(generated_text)
    
    if ground_truth_text:
        print("\nGround Truth:")
        print("-" * 80)
        print(ground_truth_text)
    
    # Save the generated text to a file
    # output_file = f"generation_{os.path.basename(args.model_path).split('.')[0]}.txt"
    # with open(output_file, 'w') as f:
    #     f.write(f"PROMPT:\n{prompt_text}\n\n")
    #     f.write(f"GENERATED:\n{generated_text}\n\n")
    #     if ground_truth_text:
    #         f.write(f"GROUND TRUTH:\n{ground_truth_text}")
    
    # print(f"\nGenerated text saved to {output_file}") 