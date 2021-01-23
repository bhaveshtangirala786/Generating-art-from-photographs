import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils import seed_everything
from dataset import ImageDataset, get_transforms
from trainer import CycleGAN
from visualize import unnorm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

''' Seeding all random parameters for reproducibility '''
seed_everything(42)

'''Default configuration for training CycleGAN'''
size = 256
batch_size = 1
mean = [0.5,0.5,0.5]
std = [0.5,0.5,0.5]

'''Reading data and creating Dataloader'''
monet_dir = 'monet_jpg/'
photo_dir = 'photo_jpg/'
transform = get_transforms(size, mean, std)
img_dataset = ImageDataset(monet_dir = monet_dir, photo_dir = photo_dir, transform = transform)
img_dl = DataLoader(img_dataset, batch_size = batch_size, pin_memory=True)

'''Fixed set of monet and photos for visualizing them throughout the training process'''
idx = np.random.randint(0, len(img_dataset), 5)
fixed_photo = torch.cat([img_dataset[i][0].unsqueeze(0) for i in idx], 0)
fixed_monet = torch.cat([img_dataset[i][1].unsqueeze(0) for i in idx], 0)

''' Creating an instance of the trainer class '''
gan = CycleGAN(3, 3, 100, device, (fixed_photo, fixed_monet), decay_epoch = 50)
gan.train(img_dl)

'''Finally visualising photos against their generated monet-style paintings''' 
fig, ax = plt.subplots(5, 2, figsize=(12, 8))
for i in range(5):
    photo_img, monet_img = next(iter(img_dl))
    pred_monet = gan.gen_ptm(photo_img.to(device)).cpu().detach()
    photo_img = unnorm(photo_img)
    pred_monet = unnorm(pred_monet)
    ax[i, 0].imshow(photo_img[0].permute(1, 2, 0))
    ax[i, 1].imshow(pred_monet[0].permute(1, 2, 0))
    ax[i, 0].set_title("Input Photo")
    ax[i, 1].set_title("Monet-gen-Portrait")
    ax[i, 0].axis("off")
    ax[i, 1].axis("off")
plt.savefig('storage/pic_to_paint.png')

    
