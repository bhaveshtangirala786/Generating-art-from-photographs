import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

''' The standard transforms for training'''
def get_transforms(size, mean, std):
    return transforms.Compose([
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)                                
            ])

''' Dataset class which samples a monet-style painting along with a photograph '''
class ImageDataset(Dataset):
    def __init__(self, monet_dir, photo_dir, transform = None):
        super().__init__()
        self.monet_dir = monet_dir
        self.photo_dir = photo_dir
        self.monet_idx = dict()
        self.photo_idx = dict()
        self.transform = transform
        
        for i, file in enumerate(os.listdir(self.monet_dir)):
            self.monet_idx[i] = file
        for i, file in enumerate(os.listdir(self.photo_dir)):
            self.photo_idx[i] = file

    def __getitem__(self, idx):
        rand_idx = int(np.random.uniform(0, len(self.monet_idx.keys())))
        photo_path = os.path.join(self.photo_dir, self.photo_idx[rand_idx])
        monet_path = os.path.join(self.monet_dir, self.monet_idx[idx])
        photo_img = Image.open(photo_path)
        photo_img = self.transform(photo_img)
        monet_img = Image.open(monet_path)
        monet_img = self.transform(monet_img)
        return photo_img, monet_img

    def __len__(self):
        return min(len(self.monet_idx.keys()), len(self.photo_idx.keys()))