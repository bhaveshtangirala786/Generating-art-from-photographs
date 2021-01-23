# Generating-art-from-photographs

Implementation of [CycleGAN](https://arxiv.org/abs/1703.10593) in PyTorch and trained on [Monet-Photo](https://www.kaggle.com/c/gan-getting-started/data) dataset

## CycleGAN
CycleGAN imposes a constraint that the model should be **cycle consistent**. If G : X -> Y transforms an image from domain X to target (Y) distribution 
and F : Y -> X transforms an image from domain Y to target (X) distribution, then cycle consistency constraint ensures that the reconstruction error is minimized
i.e **F(G(x)) ~ x**.

We also want the color distribution to be preserved, so we also impose an **identity loss**. It says that the functions G and F must be able to represent the
**Identity function**. So **G(y) ~ y** which ensures that the colors are not drastically changed when not necessary.

### Key elements of the model
 - Instance Normalization
 - Residual Blocks

## Some Training images
**Photo images**

![](pics/photos.png)

**Monet images**

![](pics/monets.png)

## Visualizing Training stages

                                                   Original Input Monet Images
![](pics/original_monet.png)

                                                   Original Input Photographs
![](pics/original_photo.png)

                                                         Epoch 0 Predicted photos
![](pics/pred_photo_0.png)

                                                        Epoch 0 Reconstructed monets
![](pics/cycle_monet_0.png)                                                               

                                                         Epoch 0 Predicted monets
![](pics/pred_monet_0.png)

                                                        Epoch 0 Reconstructed photos
![](pics/cycle_photo_0.png)

                                                         Epoch 50 Predicted photos
![](pics/pred_photo_49.png)

                                                        Epoch 50 Reconstructed monets
![](pics/cycle_monet_49.png)                                                               

                                                         Epoch 50 Predicted monets
![](pics/pred_monet_49.png)

                                                        Epoch 50 Reconstructed photos
![](pics/cycle_photo_49.png)

                                                         Epoch 100 Predicted photos
![](pics/pred_photo_99.png)

                                                        Epoch 100 Reconstructed monets
![](pics/cycle_monet_99.png)                                                               

                                                         Epoch 100 Predicted monets
![](pics/pred_monet_99.png)

                                                        Epoch 100 Reconstructed photos
![](pics/cycle_photo_99.png)
