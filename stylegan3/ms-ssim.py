from tqdm import tqdm
from ucla_protest import ProtestDataset
#https://torchmetrics.readthedocs.io/en/stable/image/multi_scale_structural_similarity.html
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)

def sampled_msssim_diversity(batch):
    ms_ssims = []
    
    #For item in batch calculate the ms-ssim of each pair
    for i in tqdm(range(len(batch))):
        for j in range(i + 1, len(batch)):
            image1 = batch[i].unsqueeze(0)
            image2 = batch[j].unsqueeze(0)
            
            ms_ssim_value = ms_ssim(image1, image2)
            ms_ssims.append(ms_ssim_value.item())
            
    # Calculate average and return 
    avg_ms_ssim = sum(ms_ssims) / len(ms_ssims)
    return avg_ms_ssim

def sampled_msssim_privacy(synth_batch, real_batch):
    ms_ssims = []

    # For each pair between synthetic and real images, calculate MS-SSIM
    for synth_image in tqdm(synth_batch):
        synth_image = synth_image.unsqueeze(0)
        for real_image in real_batch:
            real_image = real_image.unsqueeze(0)

            ms_ssim_value = ms_ssim(synth_image, real_image)
            ms_ssims.append(ms_ssim_value.item())

    # Calculate average and return
    avg_ms_ssim = sum(ms_ssims) / len(ms_ssims)
    return avg_ms_ssim
    

if __name__ == "__main__":
    

    transform = transforms.Compose([transforms.ToTensor()])

    dataset = ProtestDataset(txt_file='/scratch/UCLA-protest-test/annot_test.txt', img_dir='/localdisk0/Conditional_Generation_UCLA_November/test', multi_label=True, transform=transform)
    real_dataset = ProtestDataset(txt_file='/scratch/UCLA-protest-test/annot_test.txt', img_dir='/scratch/UCLA-protest256/img/test', multi_label=True, transform=transform)

    batch_size = 250
    runs = 5
    synth_data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    real_data_loader = DataLoader(real_dataset, batch_size=batch_size, shuffle=True)
    
    avg_ssims = []
    for i, batch in enumerate(synth_data_loader):
        real_batch = next(iter(real_data_loader))
        avg_ssims.append(sampled_msssim_privacy(batch['image'], real_batch['image']))
        if i==runs-1:
            break
    
    AVG= sum(avg_ssims) / len(avg_ssims)

    print(AVG)
    
    # Diversity Code
    
    #avg_ssims = []
    #for i, batch in enumerate(data_loader):
    #    avg_ssims.append(sampled_msssim_diversity(batch['image']))
    #   if i==runs-1:
    #        break
        
    #AVG= sum(avg_ssims) / len(avg_ssims)

    #print(AVG)