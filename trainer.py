import time
import itertools
import numpy as np
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from visualize import plot_save, unnorm
from model import Generator, Discriminator
from utils import lr_sched, AvgStats, sample_fake, init_weights, req_grad

class CycleGAN(object):
    ''' The trainer function for training two Generators which transform monet paintings to photo and vice versa; Also two Discriminators are trained to distinguish between real and fake images '''
    def __init__(self, in_channels, out_channels, epochs, device, fixed,start_lr = 2e-4, lamda = 10, idt_coef = 0.5, decay_epoch = 0):
        self.epochs = epochs
        self.decay_epoch = decay_epoch if decay_epoch > 0 else int(self.epochs / 2)
        self.lamda = lamda
        self.idt_coef = idt_coef
        self.device = device
        # transforms monet to photo
        self.gen_mtp = Generator(in_channels, out_channels)
        # transforms from photo to monet
        self.gen_ptm = Generator(in_channels, out_channels)
        self.dis_m = Discriminator(in_channels)
        self.dis_p = Discriminator(in_channels)
        self.init_models()
        self.mse_loss = nn.BCEWithLogitsLoss(reduction = "mean")
        self.l1_loss = nn.L1Loss(reduction = "mean")
        self.adam_gen = torch.optim.Adam(itertools.chain(self.gen_mtp.parameters(), self.gen_ptm.parameters()),
                                         lr = start_lr, betas=(0.5, 0.999))
        self.adam_dis = torch.optim.Adam(itertools.chain(self.dis_m.parameters(), self.dis_p.parameters()),
                                          lr=start_lr, betas=(0.5, 0.999))
        self.sample_monet = sample_fake()
        self.sample_photo = sample_fake()
        gen_lr = lr_sched(self.decay_epoch, self.epochs)
        dis_lr = lr_sched(self.decay_epoch, self.epochs)
        self.gen_lr_sched = torch.optim.lr_scheduler.LambdaLR(self.adam_gen, gen_lr.step)
        self.dis_lr_sched = torch.optim.lr_scheduler.LambdaLR(self.adam_dis, dis_lr.step)
        self.gen_stats = AvgStats()
        self.dis_stats = AvgStats()
        self.fixed_photo = fixed[0].to(self.device)
        self.fixed_monet = fixed[1].to(self.device)
        plot_save(unnorm(self.fixed_photo), 'original_photo')
        plot_save(unnorm(self.fixed_monet), 'original_monet')
        plot_save(unnorm(self.gen_mtp(self.fixed_monet)), 'pred_photo_0')
        plot_save(unnorm(self.gen_ptm(self.fixed_photo)), 'pred_monet_0')
        plot_save(unnorm(self.gen_mtp(self.gen_ptm(self.fixed_photo))), 'cycle_photo_0')
        plot_save(unnorm(self.gen_ptm(self.gen_mtp(self.fixed_monet))), 'cycle_monet_0')     
    
    # Initialising all four models        
    def init_models(self):
        init_weights(self.gen_mtp)
        init_weights(self.gen_ptm)
        init_weights(self.dis_m)
        init_weights(self.dis_p)
        self.gen_mtp = self.gen_mtp.to(self.device) # G
        self.gen_ptm = self.gen_ptm.to(self.device) # F
        self.dis_m = self.dis_m.to(self.device)
        self.dis_p = self.dis_p.to(self.device)
        
    # Training process
    def train(self, photo_dl):
        for epoch in range(self.epochs):
            start_time = time.time()
            avg_gen_loss = 0.0
            avg_dis_loss = 0.0
            if (epoch + 1)%10 == 0:
                plot_save(unnorm(self.gen_mtp(self.fixed_monet)), 'pred_photo_' + str(epoch))
                plot_save(unnorm(self.gen_ptm(self.fixed_photo)), 'pred_monet_' + str(epoch))
                plot_save(unnorm(self.gen_mtp(self.gen_ptm(self.fixed_photo))), 'cycle_photo_' + str(epoch))
                plot_save(unnorm(self.gen_ptm(self.gen_mtp(self.fixed_monet))), 'cycle_monet_' + str(epoch))
            
            for i, (photo_real, monet_real) in enumerate(tqdm(photo_dl, total = len(photo_dl))):
                photo_img, monet_img = photo_real.to(self.device), monet_real.to(self.device)
                # Ds require no gradients when optimizing Gs
                req_grad([self.dis_m, self.dis_p], False)
                self.adam_gen.zero_grad()
                
                # Training generators
                fake_photo = self.gen_mtp(monet_img)
                fake_monet = self.gen_ptm(photo_img)
                # Reconstructed images
                cycle_monet = self.gen_ptm(fake_photo)
                cycle_photo = self.gen_mtp(fake_monet)
                # Color preservation
                id_monet = self.gen_ptm(monet_img)
                id_photo = self.gen_mtp(photo_img)

                # Identity loss
                idt_loss_monet = self.l1_loss(id_monet, monet_img) * self.lamda * self.idt_coef
                idt_loss_photo = self.l1_loss(id_photo, photo_img) * self.lamda * self.idt_coef
                # Cycle consistency loss
                cycle_loss_monet = self.l1_loss(cycle_monet, monet_img) * self.lamda
                cycle_loss_photo = self.l1_loss(cycle_photo, photo_img) * self.lamda

                monet_dis = self.dis_m(fake_monet)
                photo_dis = self.dis_p(fake_photo)

                real = torch.ones(monet_dis.size()).to(self.device)
                # Adversarial loss
                adv_loss_monet = self.mse_loss(monet_dis, real)
                adv_loss_photo = self.mse_loss(photo_dis, real)

                # total generator loss
                total_gen_loss = cycle_loss_monet + adv_loss_monet + cycle_loss_photo + adv_loss_photo + idt_loss_monet + idt_loss_photo 
                avg_gen_loss += total_gen_loss.item()

                # backward pass
                total_gen_loss.backward()
                self.adam_gen.step()

                # Training disriminators
                req_grad([self.dis_m, self.dis_p], True)
                self.adam_dis.zero_grad()
                
                fake_monet = self.sample_monet([fake_monet])
                fake_photo = self.sample_photo([fake_photo])
                fake_monet = fake_monet.clone().detach()
                fake_photo = fake_photo.clone().detach()    

                monet_dis_real = self.dis_m(monet_img)
                monet_dis_fake = self.dis_m(fake_monet)
                photo_dis_real = self.dis_p(photo_img)
                photo_dis_fake = self.dis_p(fake_photo)

                real = torch.ones(monet_dis_real.size()).to(self.device)
                fake = torch.ones(monet_dis_fake.size()).to(self.device)

                # Disriminator losses
                monet_dis_real_loss = self.mse_loss(monet_dis_real, real)
                monet_dis_fake_loss = self.mse_loss(monet_dis_fake, fake)
                photo_dis_real_loss = self.mse_loss(photo_dis_real, real)
                photo_dis_fake_loss = self.mse_loss(photo_dis_fake, fake)

                monet_dis_loss = (monet_dis_real_loss + monet_dis_fake_loss) / 2
                photo_dis_loss = (photo_dis_real_loss + photo_dis_fake_loss) / 2
                total_dis_loss = monet_dis_loss + photo_dis_loss
                avg_dis_loss += total_dis_loss.item()

                # Backward
                monet_dis_loss.backward()
                photo_dis_loss.backward()
                self.adam_dis.step()
                
            self.save_checkpoint(epoch, 'storage/checkpoint.pth')
            
            avg_gen_loss /= len(photo_dl)
            avg_dis_loss /= len(photo_dl)
            time_e = time.time() - start_time
            
            self.gen_stats.append(avg_gen_loss, time_e)
            self.dis_stats.append(avg_dis_loss, time_e)
            
            print(f"Epoch: {epoch+1} | Generator Loss: {avg_gen_loss} | Discriminator Loss:{avg_dis_loss} | Time : {time_e}")
      
            self.gen_lr_sched.step()
            self.dis_lr_sched.step()

    # Saving all models and optimizers            
    def save_checkpoint(self, epoch, path):
        torch.save({
                'epoch': epoch+1,
                'gen_mtp': self.gen_mtp.state_dict(),
                'gen_ptm': self.gen_ptm.state_dict(),
                'dis_m': self.dis_m.state_dict(),
                'dis_p': self.dis_p.state_dict(),
                'optimizer_gen': self.adam_gen.state_dict(),
                'optimizer_dis': self.adam_dis.state_dict()
            }, path)
        
    # Loading all models and optimizers
    def load_checkpoint(self, path):
        ckpt = torch.load(path)
        self.gen_mtp.load_state_dict(ckpt['gen_mtp']) 
        self.gen_ptm.load_state_dict(ckpt['gen_ptm'])                     
        self.dis_m.load_state_dict(ckpt['dis_m'])                    
        self.dis_p.load_state_dict(ckpt['dis_p'])                   
        self.adam_gen.load_state_dict(ckpt['optimizer_gen'])                    
        self.adam_dis.load_state_dict(ckpt['optimizer_dis'])                 
