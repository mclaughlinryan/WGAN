import torch
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from torch import optim

import random
import torch.nn.parallel
import torch.utils.data
import torchvision.utils as vutils

import torchvision.datasets
import torchvision.transforms as transforms

# Set random seed for reproducibility
manualSeed = 999
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True,
                               transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Pad(2),
                               transforms.Normalize((0.5), (0.5)),
                            ]))

set_fraction = (3200/len(dataset)) # get fraction of data to achieve dataset of 3200 images, which is 100 batches of 32 images/batch

# image parameters for MNIST dataset
image_size = 32 # using border padded MNIST data
nc = 1

# Addition of Gaussian noise to image
class transformGaussianNoise():
    def __init__(self, mean=0, std=1):
        self.std = std
        self.mean = mean

    def __call__(self,tensor):
        return tensor + torch.randn(tensor.size())*self.std + self.mean

class transformRandomPixelErasing():
    def __init__(self, p):
        self.p = p

    def __call__(self,img):
        p_matrix = torch.full(img.size(), self.p)
        p_binary = torch.bernoulli(p_matrix).to(torch.bool)
        imgPixelErase = img.clone()
        imgPixelErase[p_binary] = -1
        return imgPixelErase

dataset_noise = torchvision.datasets.MNIST(root='./data', train=True, download=True,
                               transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Pad(2),
                               transforms.Normalize((0.5), (0.5)),
                               transformGaussianNoise(0, 0.2)
                            ]))

dataset_lr = torchvision.datasets.MNIST(root='./data', train=True, download=True,
                               transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Pad(2),
                               transforms.Resize((int)(image_size/2)),
                               transforms.Resize(image_size),
                               transforms.Normalize((0.5), (0.5))
                            ]))

dataset_erase = torchvision.datasets.MNIST(root='./data', train=True, download=True,
                               transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Pad(2),
                               transforms.Normalize((0.5), (0.5)),
                               transformRandomPixelErasing(0.6)
                            ]))

dataset, testset, _ = torch.utils.data.dataset.random_split(dataset, [int(set_fraction*len(dataset)), int((1/2)*set_fraction*len(dataset)), int(len(dataset)-(3/2)*set_fraction*len(dataset))], generator=torch.Generator().manual_seed(42))
dataset_noise, testset_noise, _ = torch.utils.data.dataset.random_split(dataset_noise, [int(set_fraction*len(dataset_noise)), int((1/2)*set_fraction*len(dataset_noise)), int(len(dataset_noise)-(3/2)*set_fraction*len(dataset_noise))], generator=torch.Generator().manual_seed(42))
dataset_lr, testset_lr, _ = torch.utils.data.dataset.random_split(dataset_lr, [int(set_fraction*len(dataset_lr)), int((1/2)*set_fraction*len(dataset_lr)), int(len(dataset_lr)-(3/2)*set_fraction*len(dataset_lr))], generator=torch.Generator().manual_seed(42))
dataset_erase, testset_erase, _ = torch.utils.data.dataset.random_split(dataset_erase, [int(set_fraction*len(dataset_erase)), int((1/2)*set_fraction*len(dataset_erase)), int(len(dataset_erase)-(3/2)*set_fraction*len(dataset_erase))], generator=torch.Generator().manual_seed(42))

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 32

# Spatial size of training images. All images will be resized to this
# size using a transformer.
image_size = 32

# Number of channels in the training images. For color images this is 3
# nc = 3
nc = 1 # Change to 1 channel if running on MNIST dataset

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps for generator
ngf = 32

# Size of feature maps for discriminator
ndf = 32

# Number of training epochs
num_epochs = 50

# Number of evaluation phase epochs
num_epochs_eval = 5

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparameter for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# Declare dataloaders
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=False, num_workers=workers)
dataloader_noise = torch.utils.data.DataLoader(dataset_noise, batch_size=batch_size,
                                         shuffle=False, num_workers=workers)
dataloader_lr = torch.utils.data.DataLoader(dataset_lr, batch_size=batch_size,
                                         shuffle=False, num_workers=workers)
dataloader_erase = torch.utils.data.DataLoader(dataset_erase, batch_size=batch_size,
                                         shuffle=False, num_workers=workers)

# Decide the device to run program on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Plot grid of training images
real_batch = next(iter(dataloader))
plt.figure(figsize=(16,16))
plt.axis("off")
plt.title("Training Images (Original dataset)")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:batch_size], padding=2, normalize=True).cpu(),(1,2,0)))

# Plot grid of Gaussian noised images
noise_batch = next(iter(dataloader_noise))
plt.figure(figsize=(16,16))
plt.axis("off")
plt.title("Training Images (Noised dataset)")
plt.imshow(np.transpose(vutils.make_grid(noise_batch[0].to(device)[:batch_size], padding=2, normalize=True).cpu(),(1,2,0)))

# Plot grid of low resolved images
lr_batch = next(iter(dataloader_lr))
plt.figure(figsize=(16,16))
plt.axis("off")
plt.title("Training Images (Low resolved dataset)")
plt.imshow(np.transpose(vutils.make_grid(lr_batch[0].to(device)[:batch_size], padding=2, normalize=True).cpu(),(1,2,0)))

# Plot grid of pixel erased images
erase_batch = next(iter(dataloader_erase))
plt.figure(figsize=(16,16))
plt.axis("off")
plt.title("Training Images (Pixel erased dataset)")
plt.imshow(np.transpose(vutils.make_grid(erase_batch[0].to(device)[:batch_size], padding=2, normalize=True).cpu(),(1,2,0)))

# custom weights initialization for generator and discriminator network
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# DCGAN generator (for use on border padded MNIST dataset)
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d( nz, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

# Create the generator
netG = Generator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# Apply weights_init function to randomly initialize all weights
# to mean=0 and stdev=0.02
netG.apply(weights_init)

# Print the model
print(netG)

# DCGAN discriminator altered for WGAN, linear output, no sigmoid activation function (for use on border padded MNIST dataset)
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False)
        )

    def forward(self, input):
        return self.main(input)

# Create the discriminator
netD = Discriminator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# Apply weights_init function to randomly initialize all weights
# to mean=0 and stdev=0.2
netD.apply(weights_init)

# Print the model
print(netD)

# Create batch of latent vectors that will be used to visualize
# progression of the generator
# Setup Adam optimizers for G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# Lists to keep track of progress
img_list_real = [[[] for idx1 in range(2)] for idx2 in range(4)]
img_list_fake = [[[] for idx1 in range(2)] for idx2 in range(4)]
G_losses_train = [[] for idx in range(4)]
G_losses_eval = [[] for idx in range(4)]
D_losses_train = [[] for idx in range(4)]
D_losses_eval = [[] for idx in range(4)]

# WGAN implementation
# Training Loop - Original data
iters = 0

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):
        # Update D network
        # Train with real batch
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        netD.zero_grad()

        # Forward pass batch through D
        real_eval = netD(real_cpu)
        errD_real = torch.mean(real_eval)

        # Calculate gradients for D in backward pass
        D_x = real_eval.mean().item()
        errD_real.backward()

        # Train with fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)

        # Generate fake image batch with G
        imgs_fake = netG(noise)

        # Classify fake batch with D
        fake_eval = netD(imgs_fake)
        errD_fake = -torch.mean(fake_eval)

        # Calculate the gradients for this batch
        D_G_z1 = fake_eval.mean().item()
        errD_fake.backward()

        # Compute error of D as sum over fake and real batches
        errD = errD_real + errD_fake

        # errD.backward()

        # Update D
        optimizerD.step()

        for p in netD.parameters():
            p.data.clamp_(-0.01, 0.01)

        if (i + 1) % 5 == 0:
            # Update G network
            netG.zero_grad()
            imgs_fake = netG(noise)

            # Since we just updated D, perform another forward pass of fake batch through D
            fake_eval = netD(imgs_fake)

            # Calculate G's loss based on this output
            errG = torch.mean(fake_eval)

            # Calculate gradients for G
            errG.backward()
            D_G_z2 = fake_eval.mean().item()

            # Update G
            optimizerG.step()

            # Save Losses to plot later
            G_losses_train[0].append(errG.item())
            D_losses_train[0].append(errD.item())

        # Keep track of generator's performance by saving output
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                img_list_real[0][0].append(vutils.make_grid(real_cpu.detach().cpu(), padding=2, normalize=True))
                img_list_fake[0][0].append(vutils.make_grid(imgs_fake.detach().cpu(), padding=2, normalize=True))

        iters += 1

    if epoch % 1 == 0:
        print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f' % (errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

# Plot real images
plt.figure(figsize=(32,32))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(img_list_real[0][0][-1],(1,2,0)))

# Plot fake images from the last epoch
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list_fake[0][0][-1],(1,2,0)))
plt.show()

# Plot of Wasserstein loss from networks over the course of training
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses_train[0],label="G")
plt.plot(D_losses_train[0],label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Evaluation Loop - Original data
iters = 0

netG.eval()
netD.eval()

print("Starting Evaluation Loop...")
# For each epoch
for epoch in range(num_epochs_eval):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):
        # Update D network
        # Train with real batch
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        # netD.zero_grad()

        # Forward pass batch through D
        real_eval = netD(real_cpu)
        errD_real = torch.mean(real_eval)

        # Calculate gradients for D in backward pass
        D_x = real_eval.mean().item()
        # errD_real.backward()

        # Train with fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)

        # Generate fake image batch with G
        imgs_fake = netG(noise)

        # Classify fake batch with D
        fake_eval = netD(imgs_fake)
        errD_fake = -torch.mean(fake_eval)

        D_G_z1 = fake_eval.mean().item()
        # errD_fake.backward()

        # Compute error of D as sum over fake and real batches
        errD = errD_real + errD_fake

        # errD.backward()

        # Update D
        # optimizerD.step()

        for p in netD.parameters():
            p.data.clamp_(-0.01, 0.01)

        if (i + 1) % 5 == 0:
            # Update G network
            # netG.zero_grad()
            imgs_fake = netG(noise)

            # Since we just updated D, perform another forward pass of fake batch through D
            fake_eval = netD(imgs_fake)

            # Calculate G's loss based on this output
            errG = torch.mean(fake_eval)

            # Calculate gradients for G
            # errG.backward()
            D_G_z2 = fake_eval.mean().item()

            # Update G
            # optimizerG.step()

            # Save Losses to plot later
            G_losses_eval[0].append(errG.item())
            D_losses_eval[0].append(errD.item())

        # Keep track of generator's performance by saving output
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                img_list_real[0][1].append(vutils.make_grid(real_cpu.detach().cpu(), padding=2, normalize=True))
                img_list_fake[0][1].append(vutils.make_grid(imgs_fake.detach().cpu(), padding=2, normalize=True))

        iters += 1

    if epoch % 1 == 0:
        print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f' % (errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

# Plotting images and loss from Evaluation phase (original dataset)
# Plot real images
plt.figure(figsize=(32,32))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(img_list_real[0][1][-1],(1,2,0)))

# Plot fake images from the last epoch
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images (Generator with no measurement function at output)")
plt.imshow(np.transpose(img_list_fake[0][1][-1],(1,2,0)))
plt.show()

# Plot of Wasserstein loss from networks during Evaluation phase
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Evaluation on Dataset Images")
plt.plot(G_losses_eval[0],label="G")
plt.plot(D_losses_eval[0],label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Network instantiation for training/evaluation phases using a pixel erasing measurement function in AmbientGAN training approach
# Create the generator
netG_erase = Generator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG_erase  = nn.DataParallel(netG_erase, list(range(ngpu)))

# Apply weights_init function to randomly initialize all weights
# to mean=0 and stdev=0.02
netG_erase.apply(weights_init)

# Create the discriminator
netD_erase = Discriminator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD_erase = nn.DataParallel(netD_erase, list(range(ngpu)))

# Apply weights_init function to randomly initialize all weights
# to mean=0 and stdev=0.2
netD_erase.apply(weights_init)

# Setup Adam optimizers for G and D
optimizerD = optim.Adam(netD_erase.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG_erase.parameters(), lr=lr, betas=(beta1, 0.999))

# Training Loop - Pixel erasing measurement function
iters = 0

class transformRandomPixelErasing():
    def __init__(self, p):
        self.p = p

    def __call__(self,img):
        p_matrix = torch.full(img.size(), self.p)
        p_binary = torch.bernoulli(p_matrix).to(torch.bool)
        imgPixelErase = img.clone()
        imgPixelErase[p_binary] = -1
        return imgPixelErase

transformPixelErasing = transforms.Compose([transformRandomPixelErasing(0.6)])

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader_erase, 0):
        # Update D network
        # Train with real batch
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        netD_erase.zero_grad()

        # Forward pass batch through D
        real_eval = netD_erase(real_cpu)
        errD_real = torch.mean(real_eval)

        # Calculate gradients for D in backward pass
        D_x = real_eval.mean().item()
        errD_real.backward()

        # Train with fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)

        # Generate fake image batch with G
        imgs_fake = netG_erase(noise)
        imgs_fake = transformPixelErasing(imgs_fake)

        # Classify fake batch with D
        fake_eval = netD_erase(imgs_fake)
        errD_fake = -torch.mean(fake_eval)

        # Calculate the gradients for this batch
        D_G_z1 = fake_eval.mean().item()
        errD_fake.backward()

        # Compute error of D as sum over fake and real batches
        errD = errD_real + errD_fake

        # errD.backward()

        # Update D
        optimizerD.step()

        for p in netD_erase.parameters():
            p.data.clamp_(-0.01, 0.01)

        if (i + 1) % 5 == 0:
            # Update G network
            netG_erase.zero_grad()
            imgs_fake = netG_erase(noise)
            imgs_fake = transformPixelErasing(imgs_fake)

            # Since we just updated D, perform another forward pass of fake batch through D
            fake_eval = netD_erase(imgs_fake)

            # Calculate G's loss based on this output
            errG = torch.mean(fake_eval)

            # Calculate gradients for G
            errG.backward()
            D_G_z2 = fake_eval.mean().item()

            # Update G
            optimizerG.step()

            # Save Losses to plot later
            G_losses_train[1].append(errG.item())
            D_losses_train[1].append(errD.item())

        # Keep track of generator's performance by saving output
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader_erase)-1)):
            with torch.no_grad():
                img_list_real[1][0].append(vutils.make_grid(real_cpu.detach().cpu(), padding=2, normalize=True))
                img_list_fake[1][0].append(vutils.make_grid(imgs_fake.detach().cpu(), padding=2, normalize=True))

        iters += 1

    if epoch % 1 == 0:
        print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f' % (errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

# Plot real images
plt.figure(figsize=(32,32))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(img_list_real[1][0][-1],(1,2,0)))

# Plot fake images from the last epoch
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list_fake[1][0][-1],(1,2,0)))
plt.show()

# Plot of Wasserstein loss from networks over the course of training
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses_train[1],label="G")
plt.plot(D_losses_train[1],label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Evaluation Loop - Pixel erasing measurement function
iters = 0

netG_erase.eval()
netD.eval()

print("Starting Evaluation Loop...")
# For each epoch
for epoch in range(num_epochs_eval):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):
        # Update D network
        # Train with real batch
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        # netD.zero_grad()

        # Forward pass batch through D
        real_eval = netD(real_cpu)
        errD_real = torch.mean(real_eval)

        # Calculate gradients for D in backward pass
        D_x = real_eval.mean().item()
        # errD_real.backward()

        # Train with fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)

        # Generate fake image batch with G
        imgs_fake = netG_erase(noise)

        # Classify fake batch with D
        fake_eval = netD(imgs_fake)
        errD_fake = -torch.mean(fake_eval)

        D_G_z1 = fake_eval.mean().item()
        # errD_fake.backward()

        # Compute error of D as sum over fake and real batches
        errD = errD_real + errD_fake

        # errD.backward()

        # Update D
        # optimizerD.step()

        for p in netD.parameters():
            p.data.clamp_(-0.01, 0.01)

        if (i + 1) % 5 == 0:
            # Update G network
            # netG.zero_grad()
            imgs_fake = netG_erase(noise)

            # Since we just updated D, perform another forward pass of fake batch through D
            fake_eval = netD(imgs_fake)

            # Calculate G's loss based on this output
            errG = torch.mean(fake_eval)

            # Calculate gradients for G
            # errG.backward()
            D_G_z2 = fake_eval.mean().item()

            # Update G
            # optimizerG.step()

            # Save Losses to plot later
            G_losses_eval[1].append(errG.item())
            D_losses_eval[1].append(errD.item())

        # Keep track of generator's performance by saving output
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                img_list_real[1][1].append(vutils.make_grid(real_cpu.detach().cpu(), padding=2, normalize=True))
                img_list_fake[1][1].append(vutils.make_grid(imgs_fake.detach().cpu(), padding=2, normalize=True))

        iters += 1

    if epoch % 1 == 0:
        print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f' % (errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

# Plotting images and loss from Evaluation phase (using pixel erased-trained network)
# Plot real images
plt.figure(figsize=(32,32))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(img_list_real[1][1][-1],(1,2,0)))

# Plot fake images from the last epoch
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images (Generator with Gaussian noise measurement function at output)")
plt.imshow(np.transpose(img_list_fake[1][1][-1],(1,2,0)))
plt.show()

# Plot of Wasserstein loss from networks during Evaluation phase
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Evaluation on Gaussian noise images")
plt.plot(G_losses_eval[1],label="G")
plt.plot(D_losses_eval[1],label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Network instantiation for training/evaluation phases using a low resolve measurement function in AmbientGAN training approach
# Create the generator
netG_lr = Generator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG_lr  = nn.DataParallel(netG_lr, list(range(ngpu)))

# Apply weights_init function to randomly initialize all weights
# to mean=0 and stdev=0.02
netG_lr.apply(weights_init)

# Create the discriminator
netD_lr = Discriminator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD_lr = nn.DataParallel(netD_lr, list(range(ngpu)))

# Apply weights_init function to randomly initialize all weights
# to mean=0 and stdev=0.2
netD_lr.apply(weights_init)

# Setup Adam optimizers for G and D
optimizerD = optim.Adam(netD_lr.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG_lr.parameters(), lr=lr, betas=(beta1, 0.999))

# Training Loop - Low resolve measurement function
iters = 0

resizeTransform_ds = transforms.Resize((int)(image_size/2))
resizeTransform_us = transforms.Resize(image_size)

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader_lr, 0):
        # Update D network
        # Train with real batch
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        netD_lr.zero_grad()

        # Forward pass batch through D
        real_eval = netD_lr(real_cpu)
        errD_real = torch.mean(real_eval)

        # Calculate gradients for D in backward pass
        D_x = real_eval.mean().item()
        errD_real.backward()

        # Train with fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)

        # Generate fake image batch with G
        imgs_fake = netG_lr(noise)
        imgs_fake = resizeTransform_ds(imgs_fake)
        imgs_fake = resizeTransform_us(imgs_fake)

        # Classify fake batch with D
        fake_eval = netD_lr(imgs_fake)
        errD_fake = -torch.mean(fake_eval)

        # Calculate the gradients for this batch
        D_G_z1 = fake_eval.mean().item()
        errD_fake.backward()

        # Compute error of D as sum over fake and real batches
        errD = errD_real + errD_fake

        # errD.backward()

        # Update D
        optimizerD.step()

        for p in netD_lr.parameters():
            p.data.clamp_(-0.01, 0.01)

        if (i + 1) % 5 == 0:
            # Update G network
            netG_lr.zero_grad()
            imgs_fake = netG(noise)
            imgs_fake = resizeTransform_ds(imgs_fake)
            imgs_fake = resizeTransform_us(imgs_fake)

            # Since we just updated D, perform another forward pass of fake batch through D
            fake_eval = netD_lr(imgs_fake)

            # Calculate G's loss based on this output
            errG = torch.mean(fake_eval)

            # Calculate gradients for G
            errG.backward()
            D_G_z2 = fake_eval.mean().item()

            # Update G
            optimizerG.step()

            # Save Losses to plot later
            G_losses_train[2].append(errG.item())
            D_losses_train[2].append(errD.item())

        # Keep track of generator's performance by saving output
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader_lr)-1)):
            with torch.no_grad():
                img_list_real[2][0].append(vutils.make_grid(real_cpu.detach().cpu(), padding=2, normalize=True))
                img_list_fake[2][0].append(vutils.make_grid(imgs_fake.detach().cpu(), padding=2, normalize=True))

        iters += 1

    if epoch % 1 == 0:
        print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f' % (errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

# Plot real images
plt.figure(figsize=(32,32))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(img_list_real[2][0][-1],(1,2,0)))

# Plot fake images from the last epoch
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list_fake[2][0][-1],(1,2,0)))
plt.show()

# Plot of Wasserstein loss from networks over the course of training
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses_train[2],label="G")
plt.plot(D_losses_train[2],label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Evaluation Loop - Low resolve measurement function
iters = 0

netG_lr.eval()
netD.eval()

print("Starting Evaluation Loop...")
# For each epoch
for epoch in range(num_epochs_eval):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):
        # Update D network
        # Train with real batch
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        # netD.zero_grad()

        # Forward pass batch through D
        real_eval = netD(real_cpu)
        errD_real = torch.mean(real_eval)

        # Calculate gradients for D in backward pass
        D_x = real_eval.mean().item()
        # errD_real.backward()

        # Train with fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)

        # Generate fake image batch with G
        imgs_fake = netG_lr(noise)

        # Classify fake batch with D
        fake_eval = netD(imgs_fake)
        errD_fake = -torch.mean(fake_eval)

        D_G_z1 = fake_eval.mean().item()
        # errD_fake.backward()

        # Compute error of D as sum over fake and real batches
        errD = errD_real + errD_fake

        # errD.backward()

        # Update D
        # optimizerD.step()

        for p in netD.parameters():
            p.data.clamp_(-0.01, 0.01)

        if (i + 1) % 5 == 0:
            # Update G network
            # netG.zero_grad()
            imgs_fake = netG_lr(noise)

            # Since we just updated D, perform another forward pass of fake batch through D
            fake_eval = netD(imgs_fake)

            # Calculate G's loss based on this output
            errG = -torch.mean(fake_eval)

            # Calculate gradients for G
            # errG.backward()
            D_G_z2 = fake_eval.mean().item()

            # Update G
            # optimizerG.step()

            # Save Losses to plot later
            G_losses_eval[2].append(errG.item())
            D_losses_eval[2].append(errD.item())

        # Keep track of generator's performance by saving output
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                img_list_real[2][1].append(vutils.make_grid(real_cpu.detach().cpu(), padding=2, normalize=True))
                img_list_fake[2][1].append(vutils.make_grid(imgs_fake.detach().cpu(), padding=2, normalize=True))

        iters += 1

    if epoch % 1 == 0:
        print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f' % (errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

# Plotting images and loss from Evaluation phase (using low resolution-trained nework)
# Plot real images
plt.figure(figsize=(32,32))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(img_list_real[2][1][-1],(1,2,0)))

# Plot fake images from the last epoch
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images (Generator with no measurement function at output)")
plt.imshow(np.transpose(img_list_fake[2][1][-1],(1,2,0)))
plt.show()

# Plot of Wasserstein loss from networks during Evaluation phase
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Evaluation on Dataset Images")
plt.plot(G_losses_eval[2],label="G")
plt.plot(D_losses_eval[2],label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Network instantiation for training/evaluation phases using a Gaussian noise measurement function in AmbientGAN training approach
# Create the generator
netG_noise = Generator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG_noise  = nn.DataParallel(netG_noise, list(range(ngpu)))

# Apply weights_init function to randomly initialize all weights
# to mean=0 and stdev=0.02
netG_noise.apply(weights_init)

# Create the discriminator
netD_noise = Discriminator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD_noise = nn.DataParallel(netD_noise, list(range(ngpu)))

# Apply weights_init function to randomly initialize all weights
# to mean=0 and stdev=0.2
netD_noise.apply(weights_init)

# Setup Adam optimizers for G and D
optimizerD = optim.Adam(netD_noise.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG_noise.parameters(), lr=lr, betas=(beta1, 0.999))

# Training Loop - Gaussian noise measurement function
iters = 0

# Addition of Gaussian noise to image
class transformGaussianNoise():
    def __init__(self, mean=0, std=1):
        self.std = std
        self.mean = mean

    def __call__(self,tensor):
        return tensor + torch.randn(tensor.size())*self.std + self.mean

transformAddNoise = transforms.Compose([transformGaussianNoise(0, 0.2)])

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader_noise, 0):
        # Update D network
        # Train with real batch
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        netD_noise.zero_grad()

        # Forward pass batch through D
        real_eval = netD_noise(real_cpu)
        errD_real = torch.mean(real_eval)

        # Calculate gradients for D in backward pass
        D_x = real_eval.mean().item()
        errD_real.backward()

        # Train with fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)

        # Generate fake image batch with G
        imgs_fake = netG_noise(noise)
        imgs_fake = transformAddNoise(imgs_fake)

        # Classify fake batch with D
        fake_eval = netD_noise(imgs_fake)
        errD_fake = -torch.mean(fake_eval)

        # Calculate the gradients for this batch
        D_G_z1 = fake_eval.mean().item()
        errD_fake.backward()

        # Compute error of D as sum over fake and real batches
        errD = errD_real + errD_fake

        # errD.backward()

        # Update D
        optimizerD.step()

        for p in netD_noise.parameters():
            p.data.clamp_(-0.01, 0.01)

        if (i + 1) % 5 == 0:
            # Update G network
            netG.zero_grad()
            imgs_fake = netG_noise(noise)
            imgs_fake = transformAddNoise(imgs_fake)

            # Since we just updated D, perform another forward pass of fake batch through D
            fake_eval = netD_noise(imgs_fake)

            # Calculate G's loss based on this output
            errG = torch.mean(fake_eval)

            # Calculate gradients for G
            errG.backward()
            D_G_z2 = fake_eval.mean().item()

            # Update G
            optimizerG.step()

            # Save Losses to plot later
            G_losses_train[3].append(errG.item())
            D_losses_train[3].append(errD.item())

        # Keep track of generator's performance by saving output
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader_noise)-1)):
            with torch.no_grad():
                img_list_real[3][0].append(vutils.make_grid(real_cpu.detach().cpu(), padding=2, normalize=True))
                img_list_fake[3][0].append(vutils.make_grid(imgs_fake.detach().cpu(), padding=2, normalize=True))

        iters += 1

    if epoch % 1 == 0:
        print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f' % (errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

# Plot real images
plt.figure(figsize=(32,32))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(img_list_real[3][0][-1],(1,2,0)))

# Plot fake images from the last epoch
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list_fake[3][0][-1],(1,2,0)))
plt.show()

# Plot of Wasserstein loss from networks over the course of training
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses_train[3],label="G")
plt.plot(D_losses_train[3],label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Evaluation Loop - Gaussian noise measurement function
iters = 0

netG_noise.eval()
netD.eval()

print("Starting Evaluation Loop...")
# For each epoch
for epoch in range(num_epochs_eval):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):
        # Update D network
        # Train with real batch
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        # netD.zero_grad()

        # Forward pass batch through D
        real_eval = netD(real_cpu)
        errD_eral = torch.mean(real_eval)

        # Calculate gradients for D in backward pass
        D_x = real_eval.mean().item()
        # errD_real.backward()

        # Train with fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)

        # Generate fake image batch with G
        imgs_fake = netG_noise(noise)

        # Classify fake batch with D
        fake_eval = netD(imgs_fake)
        errD_fake = -torch.mean(fake_eval)

        D_G_z1 = fake_eval.mean().item()
        # errD_fake.backward()

        # Compute error of D as sum over fake and real batches
        errD = errD_real + errD_fake

        # errD.backward()

        # Update D
        # optimizerD.step()

        for p in netD.parameters():
            p.data.clamp_(-0.01, 0.01)

        if (i + 1) % 5 == 0:
            # Update G network
            # netG.zero_grad()
            imgs_fake = netG_noise(noise)

            # Since we just updated D, perform another forward pass of fake batch through D
            fake_eval = netD(imgs_fake)

            # Calculate G's loss based on this output
            errG = torch.mean(fake_eval)

            # Calculate gradients for G
            # errG.backward()
            D_G_z2 = fake_eval.mean().item()

            # Update G
            # optimizerG.step()

            # Save Losses to plot later
            G_losses_train[3].append(errG.item())
            D_losses_train[3].append(errD.item())

        # Keep track of generator's performance by saving output
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                img_list_real[3][1].append(vutils.make_grid(real_cpu.detach().cpu(), padding=2, normalize=True))
                img_list_fake[3][1].append(vutils.make_grid(imgs_fake.detach().cpu(), padding=2, normalize=True))

        iters += 1

    if epoch % 1 == 0:
        print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f' % (errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

# Plotting images and loss from Evaluation phase (using Gaussian noise-trained nework)
# Plot real images
plt.figure(figsize=(32,32))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(img_list_real[3][1][-1],(1,2,0)))

# Plot fake images from the last epoch
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images (Generator with no measurement function at output)")
plt.imshow(np.transpose(img_list_fake[3][1][-1],(1,2,0)))
plt.show()

# Plot of Wasserstein loss from networks during Evaluation phase
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Evaluation on Dataset Images")
plt.plot(G_losses_eval[3],label="G")
plt.plot(D_losses_eval[3],label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()
