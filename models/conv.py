import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Normalize, Compose
from torch.utils.data import DataLoader

class ConvEncoder(nn.Module):
    def __init__(self, in_channels, token_dim):
        """
        Args:
            in_channels: Number of input channels (e.g., 3 for RGB).
            token_dim: Dimensionality of each token.
        """
        super().__init__()
        self.token_dim = token_dim
        
        # Convolutional feature extractor (2 layers)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=2, stride=2, padding=0),  # Reduce spatial dimensions
            nn.ReLU(),
            nn.Conv2d(64, token_dim, kernel_size=2, stride=2, padding=0),  # Further reduce spatial dimensions
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.conv(x)  # Shape: (batch, token_dim, H', W')
        # print('encoder:', x.shape)
        x = x.flatten(start_dim=2).transpose(1, 2)  # Reshape to (batch, n_tokens, token_dim)
        # print('encoder2:', x.shape)
        return x  # (batch, n_tokens, token_dim)

class ConvDecoder(nn.Module):
    def __init__(self, token_dim, out_channels, output_size):
        """
        Args:
            token_dim: Dimensionality of each token.
            out_channels: Number of channels in output image.
            output_size: Tuple of (H, W) for output image.
        """
        super().__init__()
        self.token_dim = token_dim
        self.output_size = output_size
        
        # Calculate the reduced dimensions after encoder's convolutions
        self.h_reduced = output_size[0] // 4  # Divided by 4 because of two stride-2 convolutions
        self.w_reduced = output_size[1] // 4

        # Deconvolutional layers to match the encoder's conv layers
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(token_dim, 64, kernel_size=2, stride=2, padding=0),  # Match encoder's second conv
            nn.ReLU(),
            nn.ConvTranspose2d(64, out_channels, kernel_size=2, stride=2, padding=0),  # Match encoder's first conv
        )

    def forward(self, x):
        batch_size = x.shape[0]
        # Reshape to match the dimensions expected by deconv
        x = x.transpose(1, 2).reshape(batch_size, self.token_dim, self.h_reduced, self.w_reduced)
        # print('decoder:', x.shape)
        x = self.deconv(x)  # Apply deconvolution
        # print('decoder2:', x.shape)
        return x

# Full Autoencoder Model
class ConvAutoencoder(nn.Module):
    def __init__(self, in_channels, token_dim, output_size):
        super().__init__()
        self.encoder = ConvEncoder(in_channels, token_dim)
        self.decoder = ConvDecoder(token_dim, in_channels, output_size)

    def forward(self, x):
        tokens = self.encoder(x)
        recon = self.decoder(tokens)
        return tokens, recon

if __name__ == "__main__":
    # Define transformations for the MNIST dataset
    transform = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,))  # Normalize with mean and std of MNIST
    ])

    # Load the MNIST dataset
    dataset = MNIST("./data", train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

    # Initialize the model
    img_channels = 1  # MNIST images are grayscale
    img_size = (28, 28)
    token_dim = 32  # 16 should be enough, but 32 is better
    model = ConvAutoencoder(in_channels=img_channels, token_dim=token_dim, output_size=img_size)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 5
    for epoch in range(num_epochs):
        for batch_idx, (data, _) in enumerate(dataloader):
            # Forward pass
            tokens, recon = model(data)
            
            # Compute loss
            loss = criterion(recon, data)

            print('loss:', loss.item())
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}')

    print("Training complete.") 