"""
Orginally created by: Donghyeon Won
"""
import os
import numpy as np
import pandas as pd
from PIL import Image

from torch.utils.data import Dataset

class ProtestDataset(Dataset):
    """
    Args:
        txt_file (str): Path to the text file containing annotations.
        img_dir (str): Directory path containing image files.
        multi_label (bool, optional): Indicates whether the dataset supports multi-label classification.
            Defaults to False.
        transform (callable, optional): A transformation function to be applied to the images.
            Defaults to None.

    Attributes:
        label_frame (DataFrame): A DataFrame containing annotations loaded from the specified text file.
        img_dir (str): The directory path containing the image files.
        img_list (list): A list of image file names in the specified directory.
        transform (callable): The transformation function to be applied to the images.
        multilabel (bool): A flag indicating whether the dataset supports multi-label classification.

    Methods:
        __len__(): Returns the number of images in the dataset.
        __getitem__(idx): Retrieves an image and its corresponding label at the given index.
    """
    def __init__(self, txt_file, img_dir, multi_label = False, transform = None):
        self.label_frame = pd.read_csv(txt_file, delimiter="\t").replace('-', 0)
        self.img_dir = img_dir
        self.img_list = sorted(os.listdir(img_dir))
        self.transform = transform
        self.multilabel = multi_label

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        imgpath = os.path.join(self.img_dir,
                                self.img_list[idx])
    
        image = pil_loader(imgpath)

        ix = self.label_frame.loc[self.label_frame['fname'] == self.img_list[idx]].index[0]

        protest = self.label_frame.iloc[ix, 1:2].to_numpy().astype('float')
        violence = (self.label_frame.iloc[ix, 2:3].to_numpy().astype('float')).astype('float')
        visattr = self.label_frame.iloc[ix, 3:].to_numpy().astype('float')
        label = {'protest':protest, 'violence':violence, 'visattr':visattr}
        if self.multilabel is True:
            label = np.concatenate((protest,violence,visattr))
        else: 
            label = np.array(protest)

        sample = {"image":image, "label":label, "fname": self.img_list[idx]}
        if self.transform:
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