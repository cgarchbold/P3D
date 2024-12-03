import argparse
import yaml
from datasets import get_dataset
from tqdm import tqdm
from models import get_model
import torch
from torch.autograd import Variable
import os
from torchvision.utils import save_image
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.kid import KernelInceptionDistance

def get_config(config):
    with open(config, 'r') as stream:
        return yaml.safe_load(stream)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #use config here
    parser.add_argument('--config', type=str, default='./configs/ucla_gan256.yaml', help='Path to the config file.')
    opts = parser.parse_args()
    config = get_config(opts.config)
    config['batch_size'] = 1
    
    # Initialize metrcs
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).cuda()
    fid = FrechetInceptionDistance(feature=2048, normalize=True).cuda()
    inception = InceptionScore(normalize=True).cuda()
    kid = KernelInceptionDistance(normalize=True).cuda()

    # Load model from checkpoint
    discriminator, generator = get_model(config)
    generator = torch.nn.DataParallel(generator, device_ids=[0])
    checkpoint = torch.load(os.path.join('./ckpts',config['experiment_name'],'model.pt'))
    generator.load_state_dict(checkpoint['g_model_state_dict'])

    train_loader, test_loader = get_dataset(config)
    
    #Create test set folder
    save_folder = os.path.join('./ckpts',config['experiment_name'],'train_set')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    #Testing Loop
    for sample in tqdm(train_loader):
        fname = sample['fname'][0]
        labels = Variable(sample['label']).cuda().type(torch.FloatTensor)
        image = Variable(sample['image']).type(torch.FloatTensor).cuda()

        # generate fake images in the range (0,1)
        z = Variable(torch.randn(image.size(0), config['noise_dim'])).cuda().float()
        fake_images = (generator(z, labels).cuda()+1.0)/2.0

        #update metrics
        #lpips.update(image, fake_images)
        inception.update(fake_images)
        fid.update(fake_images, real=False)
        fid.update(image, real=True)
        kid.update(fake_images, real=False)
        kid.update(image, real=True)
        # Save image
        save_image(fake_images[0], os.path.join(save_folder,fname))

    # Write results
    f = open(os.path.join('./ckpts',config['experiment_name'],"metrics.txt"), "a")
    f.write("FID: " + str(fid.compute().item())+'\n')
    f.write("IS: " +  str(inception.compute()[0].item())+'\n')
    #f.write("LPIPS: " + str(lpips.compute().item())+'\n')
    f.write("KID: "+ str(kid.compute()[0].item())+'\n')
    f.close()
    print('Done')