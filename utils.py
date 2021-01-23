import os
import random
import numpy as np
import torch
from torch.nn import init

''' Function to seed all random elements '''
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

''' Function to change requires_grad of all parameters of models as required '''
def req_grad(models, requires_grad = True):
    for model in models:
        for param in model.parameters():
            param.requires_grad = requires_grad
            
''' Function to initialising weights of all models as necessary '''
def init_weights(net, init_type = 'normal', gain = 0.02):
    
    def init_func(m):
        classname = m.__class__.__name__
        
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)
            
    net.apply(init_func)
    
''' Pool of previously generated images which are sampled to train the discriminators '''
class sample_fake(object):
    def __init__(self, max_imgs = 50):
        self.max_imgs = max_imgs
        self.cur_img = 0
        self.imgs = list()

    def __call__(self, imgs):
        ret = list()
        for img in imgs:
            if self.cur_img < self.max_imgs:
                self.imgs.append(img)
                ret.append(img)
                self.cur_img += 1
            else:
                # 50% chance that pool will return a stored image, and insert the current image into the pool
                if np.random.ranf() > 0.5:
                    idx = np.random.randint(0, self.max_imgs)
                    ret.append(self.imgs[idx])
                    self.imgs[idx] = img
                # other 50% chance that pool will return same image
                else:
                    ret.append(img)
        return ret[0]
    
''' Learning rate scheduler which linearly decays learning rate to zero during decay_epochs '''
class lr_sched():
    def __init__(self, decay_epochs = 100, total_epochs = 200):
        self.decay_epochs = decay_epochs
        self.total_epochs = total_epochs

    def step(self, epoch_num):
        if epoch_num <= self.decay_epochs:
            return 1.0
        else:
            fract = (epoch_num - self.decay_epochs)  / (self.total_epochs - self.decay_epochs)
            return 1.0 - fract
        
''' To keep track of all losses at each train step '''
class AvgStats(object):
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.losses =[]
        self.its = []
        
    def append(self, loss, it):
        self.losses.append(loss)
        self.its.append(it)