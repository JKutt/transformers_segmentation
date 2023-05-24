# Import of libraries
import random
import imageio
import numpy as np
from argparse import ArgumentParser

from torch.autograd import grad

from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import einops
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from torchvision.transforms import Compose, ToTensor, Lambda
from torchvision.datasets.mnist import MNIST
#from torchvision.datasets.cifar import CIFAR10
from torchModelDataLoader import modelDataset

from ddpm_networks import energyNet, UNetFlex, energyNetLS
#from ddpm_denoise import denoiseNet
# import ddpm_utils as utils

# Setting reproducibility
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Definitions
STORE_PATH_MNIST = f"ddpm_model_mnist.pt"
#STORE_PATH_MNIST = f"ddpm_model_mnist.pt"
pics = False


batch_size = 1
store_path = "ddpm_mnist.pt"


# Download data
# Loading the data (converting each image into a tensor and normalizing between [-1, 1])
# transform = Compose([
#     ToTensor(),
#     Lambda(lambda x: (x - 0.5) * 2)]
# )
# ds_fn = MNIST
# dataset = ds_fn("./datasets", download=True, train=True, transform=transform)
# loader = DataLoader(dataset, batch_size, shuffle=True)
dataset = modelDataset(directory='./models', dims=2, scale=4.11)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Getting device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\t" + (f"{torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "MPS"))

# DDPM class
class DDPM(nn.Module):
    def __init__(self, network, n_steps=200, min_beta=10 ** -4, max_beta=0.02, device=None, image_chw=(1, 32, 32)):
        super(DDPM, self).__init__()
        self.n_steps = n_steps
        self.device = device
        self.image_chw = image_chw
        self.network = network.to(device)
        self.betas = torch.linspace(min_beta, max_beta, n_steps).to(
            device)  # Number of steps is typically in the order of thousands
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.tensor([torch.prod(self.alphas[:i + 1]) for i in range(len(self.alphas))]).to(device)

    def forward(self, x0, t, eta=None):
        # Make input image more noisy (we can directly skip to the desired step)
        n, c, h, w = x0.shape
        a_bar = self.alpha_bars[t]

        if eta is None:
            eta = torch.randn(n, c, h, w).to(self.device)

        kappa = a_bar.sqrt().reshape(n, 1, 1, 1)
        std_x = (1 - a_bar).sqrt().reshape(n, 1, 1, 1)
        noisy = kappa * x0 + std_x * eta
        return noisy, kappa, std_x

    def backward(self, x, t):
        # Run each image through the network for each timestep t in the vector t.
        # The network returns its estimation of the noise that was added.
        a_bar = self.alpha_bars[t]
        #alpha_t = self.alphas[t]

        varx = (1 - a_bar).sqrt().reshape(x.shape[0], 1, 1, 1)
        score = self.network(x, t)
        
        return score
    


def integrate_backwards(ddpm, n_samples=16, device=None, frames_per_gif=100, gif_name="sampling.gif", c=1, h=32, w=32):

    """Given a DDPM model, a number of samples to be generated and a device, returns some newly generated samples"""
    frame_idxs = np.linspace(0, ddpm.n_steps, frames_per_gif).astype(np.uint)
    frames = []

    if device is None:
        device = ddpm.device

    # Starting from random noise
    # x = torch.randn(n_samples, c, h, w).to(device)
    x = torch.empty(n_samples, c, h, w).normal_(mean=0, std=2e-1).to(device)

    for idx, t in enumerate(list(range(ddpm.n_steps))[::-1]):
        # Estimating noise to be removed
        time_tensor = (torch.ones(n_samples, 1) * t).to(device).long()

        x = x.clone().detach()
        eta_theta = ddpm.backward(x, time_tensor)
        x = x.clone().detach()

        alpha_t = ddpm.alphas[t]
        alpha_t_bar = ddpm.alpha_bars[t]

        # Partially denoising the image
        x = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta)
        x = x.detach()

        if t > 0:
            # z = torch.randn(n_samples, c, h, w).to(device)
            z = torch.empty(n_samples, c, h, w).normal_(mean=0, std=2e-1).to(device)

            # Option 1: sigma_t squared = beta_t
            beta_t = ddpm.betas[t]
            sigma_t = beta_t.sqrt()

            # Adding some more noise like in Langevin Dynamics fashion
            x = x + sigma_t * z

        # Adding frames to the GIF
        if idx in frame_idxs or t == 0:
            # Putting digits in range [0, 255]
            normalized = x.clone()
            for i in range(len(normalized)):
                normalized[i] -= torch.min(normalized[i])
                normalized[i] *= 255 / torch.max(normalized[i])

            # Reshaping batch (n, c, h, w) to be a (as much as it gets) square frame
            frame = einops.rearrange(normalized, "(b1 b2) c h w -> (b1 h) (b2 w) c", b1=int(n_samples ** 0.5))
            frame = frame.cpu().numpy().astype(np.uint8)

            # Rendering frame
            frames.append(frame)

    # Storing the gif
    with imageio.get_writer(gif_name, mode="I") as writer:
        for idx, frame in enumerate(frames):
            writer.append_data(frame)
            if idx == len(frames) - 1:
                for _ in range(frames_per_gif // 3):
                    writer.append_data(frames[-1])
    return x

def training_loop(ddpm, loader, n_epochs, optim, device, display=False, store_path="ddpm_model.pt"):
    mse = nn.MSELoss()
    best_loss = float("inf")
    n_steps = ddpm.n_steps

    # determine means and cov of data
    max_hold = []
    for i, data in enumerate(loader, 0):
        
        max_hold.append(data)


    log_data = np.log(np.hstack(max_hold).flatten())
    mean_ = np.mean(log_data)
    norm_ = np.max(np.abs(log_data - mean_))

    for epoch in tqdm(range(n_epochs), desc=f"Training progress", colour="#00ff00"):
        epoch_loss  = 0.0
        epoch_lossx = 0.0
        
        for step, batch in enumerate(tqdm(loader, leave=False, desc=f"Epoch {epoch + 1}/{n_epochs}", colour="#005500")):
            # Loading data
            x0 = batch[0].to(device).view(1, 1, 128, 64)
            log_data = np.log(x0)
            norm_data = log_data - mean_
            x0 = norm_data/norm_
            # x0 = torch.nn.functional.interpolate(x0, size=[128, 64])
            n = len(x0)

            # Picking some noise for each of the images in the batch, a timestep and the respective alpha_bars
            # eta = torch.randn_like(x0).to(device)
            eta = torch.empty_like(x0).normal_(mean=0, std=2e-1).to(device)
            t = torch.randint(0, n_steps, (n,)).to(device)

            optim.zero_grad()
            # Computing the noisy image based on x0 and the time-step (forward process)
            noisy_imgs, kappa, std_x = ddpm(x0, t, eta)

            # plt.imshow(noisy_imgs.view(128, 64).cpu().detach().numpy())
            # print(t)
            # plt.show()
            
            # Getting model estimation of noise based on the images and the time-step
            noisy_imgs.requires_grad = True
            eta_theta = ddpm.backward(noisy_imgs, t.reshape(n, -1))
            SNR = kappa/std_x
            
            # z = torch.randn_like(eta_theta)
            z = torch.empty_like(eta_theta).normal_(mean=0, std=2e-1).to(device)
            z = torch.sign(z)
            f = torch.sum(z*eta_theta)
            Hz = grad(f, noisy_imgs, create_graph=True, retain_graph=True)[0]
            Tr = torch.mean(SNR*(z*Hz), dim=[1,2,3]).mean()
            # lossm = torch.mean(SNR*(eta_theta**2)) - 2*Tr
            lossm = mse(eta_theta, eta)
            
            xrec = 1/kappa*(noisy_imgs - std_x*eta_theta)
            lossx = mse(SNR*xrec, SNR*x0)/mse(x0*0, SNR*x0)
            
            
            lossm.backward()
            optim.step()

            epoch_loss  += lossm.item() * len(x0) / len(loader.dataset)
            epoch_lossx += lossx.item() * len(x0) / len(loader.dataset)
            
            #tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))

            
        log_string = f" epoch {epoch + 1}:{epoch_loss:.8f} {epoch_lossx:.8f}"

        # Storing the model
        if best_loss > epoch_loss:
            best_loss = epoch_loss
            torch.save(ddpm.state_dict(), store_path)
            log_string += " --> Best model ever (stored)"

        print(log_string)

def show_images(images, title=""):
    """Shows the provided images as sub-pictures in a square"""

    # Converting images to CPU numpy arrays
    if type(images) is torch.Tensor:
        images = images.detach().cpu().numpy()

    # Defining number of rows and columns
    fig = plt.figure(figsize=(8, 8))
    rows = int(len(images) ** (1 / 2))
    cols = round(len(images) / rows)

    # Populating figure with sub-plots
    idx = 0
    for r in range(rows):
        for c in range(cols):
            fig.add_subplot(rows, cols, idx + 1)

            if idx < len(images):
                plt.imshow(images[idx][0], cmap="Spectral_r")
                idx += 1
    fig.suptitle(title, fontsize=30)

    # Showing the figure
    plt.show()

# Training
store_path = "ddpm_mnist.pt"
# Defining model
n_steps, min_beta, max_beta = 200, 1e-4, 0.02  
nIn, nhid, nx = 1, 128, 64


# net = energyNet(nIn=nIn, nhid=nhid)
net = UNetFlex(n_steps, dims=[128,64])
#net = energyNetLS(n_steps)

ddpm = DDPM(net, n_steps=n_steps, min_beta=min_beta, max_beta=max_beta, device=device)

print('Number of parameters = ',sum([p.numel() for p in ddpm.parameters()]))

def show_forward(ddpm, loader, device):
    # determine means and cov of data
    max_hold = []
    for i, data in enumerate(loader, 0):
        
        max_hold.append(data)


    log_data = np.log(np.hstack(max_hold).flatten())
    mean_ = np.mean(log_data)
    norm_ = np.max(np.abs(log_data - mean_))
    # Showing the forward process
    eta = torch.empty(1, 1, 128, 64).normal_(mean=0, std=2e-1).to(device)
    for batch in loader:
        imgs = batch[0].view(1, 1, 128, 64)
        log_data = np.log(imgs)
        norm_data = log_data - mean_
        imgs = norm_data/norm_

        # show_images(imgs, "Original images")

        for percent in [0.25, 0.5, 0.75, 1]:
            show_images(
                ddpm(imgs.to(device),
                        [int(percent * ddpm.n_steps) - 1 for _ in range(len(imgs))], eta=eta)[0],
                f"DDPM Noisy images {int(percent * 100)}%"
            )
        break

# show_forward(ddpm, loader, device)

n_epochs = 50
lr = 1e-3

training_loop(ddpm, loader, n_epochs, optim=Adam(ddpm.parameters(), lr), device=device, store_path=store_path)

# Loading the trained model
# net_best = energyNet(nIn=nIn, nhid=nhid)  
net_best = UNetFlex(n_steps, dims=[128,64])
best_model = DDPM(net_best, n_steps=n_steps, device=device)


best_model.load_state_dict(torch.load(store_path, map_location=device))
best_model.eval()
print("Model loaded")

generated = integrate_backwards(
        best_model,
        n_samples=16,
        device=device,
        gif_name= "mnist.gif",
        h = 128,
        w = 64
    )

show_images(generated, "Final result")

print('done')
#from IPython.display import Image
#Image(open('mnist.gif','rb').read())