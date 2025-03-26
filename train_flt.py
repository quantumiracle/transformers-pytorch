from typing import Dict, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import wandb
import os
from models.diffusion_transformer import dit_mnist
from models.flow import FlowMatching

PROJECT_NAME = 'diffusion_transformer'
RUN_NAME = f'flt_mnist'
WANDB_ONLINE = True # turn this on to pipe experiment to cloud

# export cuda device
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# wandb experiment tracker
import wandb
wandb.init(project=PROJECT_NAME, mode='disabled' if not WANDB_ONLINE else 'online')
wandb.run.name = RUN_NAME
wandb.run.save()

def train_mnist():
    # hardcoding these here
    n_epoch = 20
    batch_size = 256
    n_T = 400  # number of time steps
    device = "cuda:0"
    n_classes = 10
    lrate = 1e-4
    save_model = False
    save_dir = './data/flow_outputs/'
    ws_test = [0.0, 0.5, 2.0]  # strength of generative guidance
    input_h = 28
    input_w = 28
    input_c = 1
    patch_size = 1  # critical: MNIST is 28x28, needs patch size 1
    USE_ONE_HOT_CLASS = True
    # New parameters for improved flow matching
    add_velocity_direction_loss = True
    lognorm_t = False
    target_std = 1.0  # 0.82  Target standard deviation for data shifting; disable with 1.0
    lognorm_mu = 0.0  # Mean for logit-normal distribution
    lognorm_sigma = 1.0  # Standard deviation for logit-normal distribution
    
    if USE_ONE_HOT_CLASS:
        external_cond_dim = 10  # MNIST has 10 classes
    else:
        external_cond_dim = 1
    os.makedirs(save_dir, exist_ok=True)

    # load model
    model = dit_mnist(input_h=input_h, input_w=input_w, in_channels=input_c, 
                     patch_size=patch_size, external_cond_dim=external_cond_dim)
    flow_model = FlowMatching(model, n_T=n_T, device=device, drop_prob=0.1,
                             add_velocity_direction_loss=add_velocity_direction_loss,
                             lognorm_t=lognorm_t, target_std=target_std,
                             lognorm_mu=lognorm_mu, lognorm_sigma=lognorm_sigma)
    flow_model.to(device)

    # optionally load a model
    # flow_model.load_state_dict(torch.load("./data/flow_outputs/flow_model_19.pth"))

    tf = transforms.Compose([transforms.ToTensor()])  # mnist is already normalised 0 to 1

    dataset = MNIST("./data", train=True, download=True, transform=tf)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=5)
    optim = torch.optim.Adam(flow_model.parameters(), lr=lrate)

    for ep in range(n_epoch):
        print(f'epoch {ep}')
        flow_model.train()

        # linear lrate decay
        optim.param_groups[0]['lr'] = lrate * (1 - ep / n_epoch)

        pbar = tqdm(dataloader)
        loss_ema = None
        for x, c in pbar:
            optim.zero_grad()
            x = x.to(device)
            c = c.to(device)
            
            # Generate random noise as the starting point
            x_noise = torch.randn_like(x).to(device)
            
            # one-hot encoding
            if USE_ONE_HOT_CLASS:
                one_hot_c = torch.zeros(c.size(0), n_classes).to(device)
                one_hot_c.scatter_(1, c.unsqueeze(1), 1)
            else:
                one_hot_c = c.unsqueeze(1)
                
            # Train flow matching from noise to data
            loss = flow_model(x_noise, x, one_hot_c)
            loss.backward()
            
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()
            wandb.log({"loss": loss_ema})
            
        # for eval, save an image of currently generated samples (top rows)
        # followed by real images (bottom rows)
        flow_model.eval()
        with torch.no_grad():
            n_sample = 4 * n_classes
            for w_i, w in enumerate(ws_test):
                # get condition
                int_c_i = torch.arange(0, 10).to(device)  # context cycles through mnist labels
                if USE_ONE_HOT_CLASS:
                    c_i = torch.zeros(int_c_i.shape[0], 10).to(device)
                    c_i.scatter_(1, int_c_i.unsqueeze(1), 1)
                else:
                    c_i = int_c_i.unsqueeze(1)
                c_i = c_i.repeat(int(n_sample / c_i.shape[0]), 1)
                
                # Sample using flow matching
                x_gen, x_gen_store = flow_model.sample(n_sample, (input_c, input_h, input_w), 
                                                     device, guide_w=w, cond=c_i)

                # append some real images at bottom, order by class also
                x_real = torch.Tensor(x_gen.shape).to(device)
                for k in range(n_classes):
                    for j in range(int(n_sample / n_classes)):
                        try:
                            idx = torch.squeeze((c == k).nonzero())[j]
                        except:
                            idx = 0
                        x_real[k + (j * n_classes)] = x[idx]

                x_all = torch.cat([x_gen, x_real])
                grid = make_grid(x_all * -1 + 1, nrow=10)
                save_image(grid, save_dir + f"image_ep{ep}_w{w}.png")
                print('saved image at ' + save_dir + f"image_ep{ep}_w{w}.png")

                if ep % 5 == 0 or ep == int(n_epoch - 1):
                    # create gif of images evolving over time, based on x_gen_store
                    fig, axs = plt.subplots(nrows=int(n_sample / n_classes), ncols=n_classes, 
                                           sharex=True, sharey=True, figsize=(8, 3))
                    def animate_flow(i, x_gen_store):
                        print(f'gif animating frame {i} of {x_gen_store.shape[0]}', end='\r')
                        plots = []
                        for row in range(int(n_sample / n_classes)):
                            for col in range(n_classes):
                                axs[row, col].clear()
                                axs[row, col].set_xticks([])
                                axs[row, col].set_yticks([])
                                plots.append(axs[row, col].imshow(-x_gen_store[i, (row * n_classes) + col, 0], 
                                                                cmap='gray', 
                                                                vmin=(-x_gen_store[i]).min(), 
                                                                vmax=(-x_gen_store[i]).max()))
                        return plots
                    
                    ani = FuncAnimation(fig, animate_flow, fargs=[x_gen_store], interval=200, 
                                       blit=False, repeat=True, frames=x_gen_store.shape[0])
                    ani.save(save_dir + f"gif_ep{ep}_w{w}.gif", dpi=100, writer=PillowWriter(fps=5))
                    wandb.log({f"gif_ep{ep}_w{w}.gif": wandb.Video(save_dir + f"gif_ep{ep}_w{w}.gif")})
                    print('saved image at ' + save_dir + f"gif_ep{ep}_w{w}.gif")
                    
        # optionally save model
        if save_model and ep == int(n_epoch - 1):
            torch.save(flow_model.state_dict(), save_dir + f"model_{ep}.pth")
            print('saved model at ' + save_dir + f"model_{ep}.pth")

if __name__ == "__main__":
    train_mnist()
