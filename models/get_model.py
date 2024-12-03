from .conv32 import CNNDiscriminator, CNNGenerator
from .conv256 import CNNDiscriminator256, CNNGenerator256
from .resnet256 import ResNetGenerator, ResNetDiscriminator
from .resnet32 import ResGenerator32, ResDiscriminator32
import torch 

'''
    Generator models must have tanh output
'''

def get_model(config):
    if config['model'] == 'conv32':
        if config['conditional'] == True:
            generator = CNNGenerator(z_dim = config['noise_dim'], channels = config['input_channels'], multi_label=config['multi_label'], num_labels=config['num_labels']).cuda()
            discriminator = CNNDiscriminator(channels = config['input_channels'], multi_label=config['multi_label'], num_labels=config['num_labels']).cuda()
        else:
            generator = CNNGenerator(z_dim = config['noise_dim'], num_classes = None, channels = config['input_channels']).cuda()
            discriminator = CNNDiscriminator(num_classes = None, channels = config['input_channels']).cuda()

    if config['model'] == 'conv256':
        if config['conditional'] == True:
            generator = CNNGenerator256(z_dim = config['noise_dim'], num_classes=config['num_classes'], channels = config['input_channels']).cuda()
            discriminator = CNNDiscriminator256(channels = config['input_channels']).cuda()
        else:
            generator = CNNGenerator256(z_dim = config['noise_dim'], num_classes = None, channels = config['input_channels']).cuda()
            discriminator = CNNDiscriminator256(num_classes = None, channels = config['input_channels']).cuda()

    if config['model'] == 'ResNet256':
        if config['conditional'] == True:
            generator = None
            discriminator = None
        else:
            generator = ResNetGenerator(noise_dim = config['noise_dim']).cuda()
            discriminator = ResNetDiscriminator().cuda()

    if config['model'] == 'ResNet32':
        if config['conditional'] == True:
            generator = ResGenerator32(z_dim = config['noise_dim'], multi_label=config['multi_label'], num_labels=config['num_labels']).cuda()
            discriminator = ResDiscriminator32(multi_label=config['multi_label'], num_labels=config['num_labels']).cuda()
        else:
            generator = ResGenerator32(z_dim = config['noise_dim'],num_classes = None).cuda()
            discriminator = ResDiscriminator32(num_classes = None).cuda()



    return discriminator, generator