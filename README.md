# WGAN

This program uses a GAN with a Wasserstein loss function to train on image data that has had some type of function applied to it resulting in data loss (erased pixels, lowered resolution, or added noise). During training, the function causing data loss is applied to the output of the GAN generator and the discriminator must discern between such output and the images with data loss. This setup induces the generator to learn the underlying true data distribution of the images to best imitate the images with data loss subject to the data loss function being applied to its output.

This program was written with PyTorch, and the model training is done with the MNIST dataset using batches of images with 32 images per batch. An Adam optimizer is used.
