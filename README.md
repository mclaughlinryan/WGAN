# WGAN

This program was written with PyTorch and uses a GAN with a Wasserstein loss function to train on image data that has had some type of data loss function applied to it (erased pixels, lowered resolution, or added noise). During training, the function causing data loss is applied to the output of the GAN generator and the discriminator must discern between such output and real images with data loss. This setup induces the generator to learn the underlying true data distribution of the images to best imitate real images with data loss subject to the data loss function being applied to its output. The model training is done with the MNIST dataset using batches of images with 32 images per batch and an Adam optimizer as the optimizer used.

### Training on MNIST images

MNIST images:
