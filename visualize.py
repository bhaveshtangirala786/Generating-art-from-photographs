import torch
import matplotlib.pyplot as plt

''' Function to un-normalize tensors to images '''
def unnorm(img, mean = [0.5,0.5,0.5], std = [0.5,0.5,0.5]):
    img1 = img.clone().detach().cpu()
    s = torch.tensor(std).view(-1,1,1)
    m = torch.tensor(mean).view(-1,1,1)
    for i in range(img1.shape[0]):
        img1[i] = (img1[i] * s) + m
    return img1

''' Function to plot Generator and Discriminator losses '''
def plot_loss(gen_loss, dis_loss):
    plt.xlabel("Epochs")
    plt.ylabel("Losses")
    plt.plot(gen_loss, 'r', label='Generator Loss')
    plt.plot(dis_loss, 'b', label='Discriminator Loss')
    plt.legend()
    plt.savefig('storage/losses.png')
    plt.show()
    
''' Function to save generated images for visualisation '''
def plot_save(imgs, path):
    fig, ax = plt.subplots(1, len(imgs), figsize=(15, 5))
    for i in range(len(imgs)):
        ax[i].imshow(imgs[i].permute(1, 2, 0))
        ax[i].axis("off")
    plt.savefig('storage/viz/' + path + '.png')
    plt.close()
    