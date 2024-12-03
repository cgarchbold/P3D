import torch
from torch.autograd import Variable
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import random

# For plotting MNIST and CIFAR samples
def plot_MNIST_samples(generator, path, config):
    z = Variable(torch.randn(100, config['noise_dim'])).cuda()
    #labels = torch.LongTensor([i for i in range(10) for _ in range(10)]).cuda()
    labels = torch.LongTensor([[random.choice([0, 1]) for _ in range(12)] for _ in range(100)]).cuda()

    if config['conditional'] == True:
        images = generator(z, labels).cpu()
    else:
        images = generator(z).cpu()

    grid = make_grid(images, nrow=10, normalize=True)

    fig, ax = plt.subplots(figsize=(20,20))
    ax.imshow(grid.permute(1, 2, 0).data, cmap='binary')
    ax.axis('off')
    plt.savefig(path)
    plt.close()

#TODO
def plot_ucla_samples(data, path):
    grid = make_grid(data, nrow=8, normalize=True)

    #TODO: I want to plot 5x4 (20) postitive protest images, and 5x4(20) negative protest images

    fig, ax = plt.subplots(figsize=(20,20))

    plt.imshow(grid.permute(1,2,0))
    ax.axis('off')
    plt.savefig(path)
    plt.close()