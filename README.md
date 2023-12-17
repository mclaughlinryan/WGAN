# WGAN

This program was written with PyTorch and uses a GAN with a Wasserstein loss function to train on image data that has had some type of data loss function applied to it (erased pixels, lowered resolution, or added noise). During training, the function causing data loss is applied to the output of the GAN generator and the discriminator must discern between such output and real images with data loss. This setup induces the generator to learn the underlying true data distribution of the images to best imitate real images with data loss subject to the data loss function being applied to its output. The model training is done with the MNIST dataset using batches of images with 32 images per batch and an Adam optimizer as the optimizer used.

### Training on MNIST images

MNIST images:

<img width="800" alt="wgan 5 1" src="https://github.com/mclaughlinryan/WGAN/assets/150348966/b9c7221c-2a43-425c-98d6-e5bed8cee84a">

&nbsp;

Output images from the generator:

<img width="800" alt="wgan 5 2" src="https://github.com/mclaughlinryan/WGAN/assets/150348966/157fb4ab-428a-45e7-9a17-48b6f147ce87">

### Training to recover pixel-erased image data

Pixel-erased MNIST images:

<img width="800" alt="wgan 9 1" src="https://github.com/mclaughlinryan/WGAN/assets/150348966/f3168d32-90f8-4b1b-abd3-6c54817ef736">

&nbsp;

Images after applying pixel erasing to the generator output:

<img width="800" alt="wgan 9 2" src="https://github.com/mclaughlinryan/WGAN/assets/150348966/0b24d89b-26c0-492e-9416-e87d9a8be929">
