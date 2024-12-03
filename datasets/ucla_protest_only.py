"""
created by: Donghyeon Won
"""

import os
import numpy as np
import pandas as pd
from PIL import Image

from torch.utils.data import Dataset
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models


class ProtestDatasetONLY(Dataset):
    """
    dataset for training and evaluation
    """
    def __init__(self, txt_file, img_dir, transform = None):
        """
        Args:
            txt_file: Path to txt file with annotation
            img_dir: Directory with images
            transform: Optional transform to be applied on a sample.
        """
        self.label_frame = pd.read_csv(txt_file, delimiter="\t").replace('-', 0)
        
        self.img_dir = img_dir
        self.img_list = sorted(os.listdir(img_dir))
        
        # Grabbing only protest imagery, and removing missing samples
        new_list = []
        for path in self.img_list:
            if self.label_frame.loc[self.label_frame['fname'] == path]['protest'].values == 1:
                #remove from list
                new_list.append(path)
            
        self.img_list = new_list
        self.transform = transform

    def __len__(self):
        return len(self.img_list)
    def __getitem__(self, idx):
        imgpath = os.path.join(self.img_dir,
                                self.img_list[idx])
        
        image = pil_loader(imgpath)

        if self.transform:
            image = self.transform(image)
        return image, 1, self.img_list[idx]

class ProtestDatasetEvalONLY(Dataset):
    """
    dataset for just calculating the output (does not need an annotation file)
    """
    def __init__(self, txt_file, img_dir):
        """
        Args:
            img_dir: Directory with images
        """
        self.img_dir = img_dir
        self.transform = transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225]),
                                ])
        self.img_list = sorted(os.listdir(img_dir))
        self.label_frame = pd.read_csv(txt_file, delimiter="\t").replace('-', 0)
    def __len__(self):
        return len(self.label_frame)
    def __getitem__(self, idx):
        imgpath = os.path.join(self.img_dir,
                                self.label_frame.iloc[idx, 0])
        image = pil_loader(imgpath)
        # we need this variable to check if the image is protest or not)
        sample = {"imgpath":imgpath, "image":image}
        sample["image"] = self.transform(sample["image"])
        return sample
    

def pil_loader(path): 
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    try:
        with open(path, 'rb') as f:
            img = Image.open(path)
            #print("Path Found: "+ path)
            return img.convert('RGB')
    except:
        print("Path Missing: "+ path)
        return Image.new(mode="RGB", size=(100,100))