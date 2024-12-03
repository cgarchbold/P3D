import argparse
import yaml
import wandb
import os
import torch

from datasets import get_dataset
from models import get_model
from util import calculate_sigma
from trainers import train_wgan

def get_config(config_pt):
    """
    Load a YAML configuration file and return its content as a Python dictionary.
    """
    with open(config_pt, 'r') as stream:
        return yaml.safe_load(stream)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/ucla_gan256.yaml', help='Path to the config file.')
    opts = parser.parse_args()
    config = get_config(opts.config)

    os.environ["CUDA_VISIBLE_DEVICES"] = config['cuda_devices']

    #Data choice
    train_loader, test_loader = get_dataset(config)

    print("Train length ", len(train_loader))
    print("Test length ", len(test_loader))

    if config['sigma'] == -1:
        config['sigma'] = calculate_sigma(config['epsilon'], config['n_critic'],config['iterations'], len(train_loader), config['privacy_violation'] )
        print("Calculated Sigma: ", config['sigma'])

    if config['wandb']==True:   
        wandb.init(
        # set the wandb project where this run will be logged
        entity='chill-cga',
        project="dp-protest",
        group= config['model'],
        name = str( config['experiment_name']) + ' sigma: ' + str(config['sigma']) + ' wc: '+ str(config['weight_clip']),
        # track hyperparameters and run metadata
        config = config
        )

    #Model choice
    discriminator, generator = get_model(config)

    #TODO: Load from checkpoint, if resume
    
    discriminator = torch.nn.DataParallel(discriminator, device_ids=[0])
    generator = torch.nn.DataParallel(generator, device_ids=[1])

    #Parameter Counts
    G_total_params = sum(p.numel() for p in generator.parameters())
    print(f"Generator: Number of parameters: {G_total_params}")

    D_total_params = sum(p.numel() for p in discriminator.parameters())
    print(f"Discriminator: Number of parameters: {D_total_params}")

    #Optimizer choice
    if config['optim'] == 'adamw':
        d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=config['d_learning_rate'], weight_decay = config['weight_decay'])
        g_optimizer = torch.optim.Adam(generator.parameters(), lr=config['g_learning_rate'], weight_decay = config['weight_decay'])
    if config['optim'] == 'rmsprop':
        d_optimizer = torch.optim.RMSprop(discriminator.parameters(), lr=config['d_learning_rate'], weight_decay = config['weight_decay'])
        g_optimizer = torch.optim.RMSprop(generator.parameters(), lr=config['g_learning_rate'], weight_decay = config['weight_decay'])

    train_wgan(config, discriminator, generator, d_optimizer, g_optimizer, train_loader, test_loader)
    #TODO: Run test code
    #TODO: Save config