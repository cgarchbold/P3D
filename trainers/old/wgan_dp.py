import torch
import numpy as np
from torch.autograd import Variable
from torchvision.utils import make_grid
import wandb
from tqdm import tqdm
import notebooks.plotting as plotting
import os
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore


"""
    Wasserstein GAN G train step
"""
def generator_train_step(config, noise_dim, batch_size, discriminator, generator, g_optimizer):
    g_optimizer.zero_grad()
    z = Variable(torch.randn(batch_size, noise_dim)).cuda()
    fake_labels = Variable(torch.LongTensor(np.random.randint(0, 10, batch_size))).cuda()

    if config['conditional'] == True:
        fake_images = generator(z, fake_labels)
        validity = discriminator(fake_images, fake_labels)
    else:
        fake_images = generator(z)
        validity = discriminator(fake_images)

    g_loss = -torch.mean(validity)
    g_loss.backward()
    g_optimizer.step()
    return g_loss.item()

"""
    Wasserstein GAN D train step
"""
def discriminator_train_step(config, noise_dim, batch_size, discriminator, generator, d_optimizer, real_images, labels, weight_clip=None):
    d_optimizer.zero_grad()

    # train with real images
    if config['conditional'] == True:
        real_validity = discriminator(real_images, labels)
    else:
        real_validity = discriminator(real_images)

    real_loss = -torch.mean(real_validity)
    
    # train with fake images
    z = Variable(torch.randn(batch_size, noise_dim)).cuda()

    # TODO: Conditional Setup
    fake_labels = Variable(torch.LongTensor(np.random.randint(0, 10, batch_size))).cuda()

    if config['conditional'] == True:
        fake_images = generator(z, fake_labels)
        fake_validity = discriminator(fake_images, fake_labels)
    else:
        fake_images = generator(z)
        fake_validity = discriminator(fake_images)

    fake_loss = torch.mean(fake_validity)
    
    d_loss = real_loss + fake_loss
    d_loss.backward()
    d_optimizer.step()

    #Weight clipping for privacy guarantee
    if weight_clip is not None:
        for param in discriminator.parameters():
            param.data.clamp_(-weight_clip, weight_clip)

    return d_loss.item()


"""
    Trains a conditional Wasserstein GAN with weight clipping + noise addition
"""
def train_wgan_dp(config, discriminator, generator, d_optimizer, g_optimizer, train_loader, test_loader):

    if config['sigma'] is not None:
        for parameter in discriminator.parameters():
            def add_noise_to_gradient(grad):
                noise = (1 / config['batch_size']) * config['sigma'] * torch.randn(grad.shape).cuda()
                return grad + noise

            parameter.register_hook(add_noise_to_gradient)

    min_val = 9999999999999999999
    for i in tqdm(range(config['iterations'])):
            
        step = i+1 
        generator.train()

        # Discriminator step
        d_loss = 0
        for _ in range(config['n_critic']):
            images,labels = next(iter(train_loader))
            real_images = Variable(images).cuda()
            labels = Variable(labels).cuda()
            d_loss += discriminator_train_step(config, config['noise_dim'], config['batch_size'], discriminator,
                                            generator, d_optimizer,
                                            real_images, labels, config['weight_clip'] )
            

        # Generator Step
        g_loss = generator_train_step(config, config['noise_dim'], config['batch_size'], discriminator, generator, g_optimizer)
            
        if config['wandb'] == True:
            wandb.log({"g_loss": g_loss, "d_loss": d_loss/config['n_critic'] })
            
        if step % config['display_step']  == 0 and config['wandb'] == True:
            generator.eval()
            z = Variable(torch.randn(9, config['noise_dim'])).cuda()
            labels = Variable(torch.LongTensor(np.arange(9))).cuda()

            if config['conditional'] == True:
                sample_images = generator(z, labels)#.unsqueeze(1)
            else:
                sample_images = generator(z)#.unsqueeze(1)
            grid = make_grid(sample_images, nrow=3, normalize=True)
            images = wandb.Image(
                grid.cpu(), 
                caption=""
                )
            
            wandb.log({'sample_images': images})
        #Train Loop end

        #TODO:Move to wandb
        if i + 1 == 1000 and config['dataset'] == 'MNIST':
            plotting.plot_MNIST_samples(generator, './plots/'+str(config['experiment_name'])+'_1000.png', config)
        if i + 1 == 10000 and config['dataset'] == 'MNIST':
            plotting.plot_MNIST_samples(generator, './plots/'+str(config['experiment_name'])+'_10000.png', config)

        #TODO: Test Step: For now the last iteration
        if  i + 1 == config['iterations']:
            print("Testing")

            fid = FrechetInceptionDistance(normalize = True).cuda()
            inception = InceptionScore(normalize = True).cuda()
            
            if config['dataset'] == 'MNIST' or config['dataset'] == 'CIFAR10':
                plotting.plot_MNIST_samples(generator, './plots/'+str(config['experiment_name'])+'_final.png', config)

            for i, (images, labels) in tqdm(enumerate(test_loader)):
                z = Variable(torch.randn(labels.size(0), config['noise_dim'])).cuda()

                real_images = Variable(images).cuda()
                #labels = Variable(labels).cuda()
                
                #Normalize to 0-1
                if config['conditional']:
                    imgs = (generator(z, labels).squeeze(0) + 1.0 )/ 2.0
                else:
                    imgs = (generator(z).squeeze(0) + 1.0) / 2.0

                # Consider the MNIST Case
                if config['input_channels'] != 3:
                    imgs = torch.cat([imgs, imgs, imgs], dim=1)
                    real_images = torch.cat([real_images, real_images, real_images], dim=1)

                fid.update(real_images.cuda(), real=True)
                fid.update(imgs, real=False)
                inception.update(imgs)

            inc_mean, inc_std = inception.compute()
            mean_fid = fid.compute().cpu()
            
            #Save last model (test model)
            path = os.path.join('./ckpts/',config['experiment_name']+str(min_val)+'.pt')
            torch.save(generator.state_dict(), path)
            print('Saving Model')
                    
            if config['wandb'] == True:
                my_table = wandb.Table(columns=["IS_mean","IS_std", "FID_mean",'W1'], data=[[inc_mean,inc_std,mean_fid, g_loss]])
                wandb.log({"test_results": my_table})

    print('Done!')