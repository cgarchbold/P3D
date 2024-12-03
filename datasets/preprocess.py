import os
import torchvision.transforms as transforms
from tqdm import tqdm
from ucla_protest import ProtestDataset

# Function to save image from a dataset
def resize(dataset, path):

    for sample in tqdm(dataset):
        image = sample['image']
        image.save(os.path.join(path,sample['fname']))

#This script loads the UCLA dataset, and saves a copy resized at the size parameter
if __name__ == '__main__':
    im_size = 32

    root_dir = "/scratch/UCLA-protest/"
    out_dir = "/scratch/UCLA-protest32/"

    train_transform =  train_transform = transforms.Compose([
            transforms.Resize(size = (im_size,im_size))
        ])
    
    test_transform =  train_transform = transforms.Compose([
            transforms.Resize(size = (im_size,im_size)),
        ])
    train_dataset = ProtestDataset(
                        txt_file = os.path.join(root_dir, 'annot_train.txt'),
                        img_dir = os.path.join(root_dir, 'img', 'train'),
                        transform = train_transform)
    
    test_dataset = ProtestDataset(
                        txt_file = os.path.join(root_dir, 'annot_test.txt'),
                        img_dir = os.path.join(root_dir,'img','test'),
                        transform= test_transform)
    
    # Save new dataset
    print("Preprocessing Train set...")
    resize(train_dataset,os.path.join(out_dir,'img','train'))
    print("Preprocessing Test set...")
    resize(test_dataset,os.path.join(out_dir,'img','test'))