#!/usr/bin/env python
# coding: utf-8

# Importing Libraries:
# ==

# In[2]:



import time
from tqdm.notebook import tqdm
import numpy as np
import matplotlib.pyplot as plt
from livelossplot import PlotLosses


import torch
import torch.nn as nn
from torch import optim
from torch.utils import data
from torchvision.transforms import functional as F
import torchvision
import shutil

from sklearn.metrics import jaccard_score as jsc

from PIL import Image
from matplotlib.path import Path
import os
import pandas as pd

import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

use_gpu = torch.cuda.is_available()
if use_gpu:
    print('GPU is available!')
    device = "cuda"
else:
    print('GPU is not available!')
    device = "cpu"


# Hyperparams
# ==

# In[3]:


pretrained = True
learning_rate = 0
test_size = 25
img_size = 256


# Loading Data:
# ==

# In[4]:


def transform(x):
    return  torchvision.transforms.functional.to_tensor(x)

class DataSet(torch.utils.data.Dataset):
    def __init__(self, mode):
        super().__init__()
        self.image_names= os.listdir('./Data_ML/image_chips/')
        self.mode = mode
    def __getitem__(self, index):
        image = plt.imread(os.path.join(os.getcwd(),'mosaic_test.jpg'))
        x,y = (index)//5  , (index+1)%5
        if(y==0):   
            y=5
        img_crop=image[(x)*750:(x+1)*750,(y-1)*750:(y)*750,:]
        return torchvision.transforms.Resize((img_size,img_size))(transform(img_crop))
    def __len__(self):
        return test_size


# In[5]:


input_channels = 3 
output_classes = 1

test_dataLoader = data.DataLoader(
    DataSet('Test'),
    batch_size=1,
    shuffle=False,
    num_workers=0
)


# Creating Module:
# ==

# In[12]:


class UNetConv(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.conv2 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv2(x)
class DownConv(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.conv_down = nn.Sequential(
            nn.MaxPool2d(2, 2),
            UNetConv(input_channels, output_channels)
        )
    def forward(self, x):
        return self.conv_down(x)
class UpConv(nn.Module): 
    def __init__(self, input_channels, output_channels, padding=0):
        super().__init__()
        self.conv_up = nn.Sequential(
            nn.ConvTranspose2d(input_channels , input_channels // 2, kernel_size=2, stride=2, padding = padding),
        )
        self.conv_level = nn.Sequential(
            UNetConv(input_channels,output_channels) 
        )
    def forward(self, x1, x2):
        return self.conv_level(torch.cat([x2,self.conv_up(x1)],dim = 1)) 
class LastConv(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.conv_final = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=1)
        )
    def forward(self, x):
        return self.conv_final(x)
class UNet(nn.Module):
    def __init__(self, input_channels, min_channels):
        super(UNet, self).__init__()
        self.current_channels = min_channels
        self.start = UNetConv(input_channels, self.current_channels)
        self.current_channels = self.current_channels*2
        self.down1 = DownConv(self.current_channels // 2 , self.current_channels)
        self.current_channels = self.current_channels*2
        self.down2 = DownConv(self.current_channels // 2 , self.current_channels)
        self.current_channels = self.current_channels*2
        self.down3 = DownConv(self.current_channels // 2 , self.current_channels)
        self.current_channels = self.current_channels*2
        self.down4 = DownConv(self.current_channels // 2 , self.current_channels)
        self.current_channels = self.current_channels // 2
        self.up1 = UpConv(self.current_channels * 2, self.current_channels) 
        self.current_channels = self.current_channels // 2
        self.up2 = UpConv(self.current_channels * 2, self.current_channels)
        self.current_channels = self.current_channels // 2
        self.up3 = UpConv(self.current_channels * 2, self.current_channels)
        self.current_channels = self.current_channels // 2
        self.up4 = UpConv(self.current_channels * 2, self.current_channels)
        self.final = LastConv(self.current_channels,output_classes)
    def forward(self, x):
        x1 = self.start(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        y1 = self.up1(x5,x4)
        y2 = self.up2(x4,x3) 
        y3 = self.up3(y2,x2) 
        x = self.up4(y3,x1)
        x = torch.nn.Dropout2d(p=dropout_percent/100)(x)
        return torch.sigmoid(self.final(x))

if(pretrained):
    Model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=3, out_channels=1, init_features=32, pretrained=True)
else:
    Model = UNet(input_channels, min_channels)
    Model = Model.to(device)

if(use_gpu):
    Model = nn.DataParallel(Model, device_ids = [i for i in range(torch.cuda.device_count())]).to(0)


optimizer = optim.Adam(Model.parameters(),lr=learning_rate)


# In[13]:


def load_ckp(checkpoint_fpath, model, optimizer):
    if(use_gpu==False):
        checkpoint = torch.load(checkpoint_fpath,map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    loss = checkpoint['loss']
    acc = checkpoint['accuracy']
    return model, optimizer, checkpoint['epoch'], loss, acc


# Testing Model:
# ==

# In[11]:


names = ['9thRealJob100','8thRealJob100','4thRealJob100','2ndRealJob100','5thRealJob100']
loc = 'EnsembleResultsBias'
index = 0
ans = np.ones((3750,3750))
for inputs in tqdm(test_dataLoader):
    inputs = inputs.to(device)
    if(use_gpu):
        inputs = inputs.to(0)
    foutputs = np.zeros((750,750))
    for name in names:
        PATH = os.path.join(os.getcwd(),'models/',name+'.pt')
        bestPATH = os.path.join(os.getcwd(),'models/best_model_'+name+'.pt')
        Model, optimizer, epoch, loss, acc = load_ckp(bestPATH,Model,optimizer) 
        Model.eval()
        outputs = Model(inputs) 
        outputs = torchvision.transforms.Resize((750,750))(outputs)
        outputs = outputs.cpu().detach().numpy().reshape((750,750))
        foutputs += outputs
        #foutputs += (list(map(lambda x : [i**2 for i in x],outputs)))
    foutputs/=len(names)
    #foutputs = (list(map(lambda x : [i**0.5 for i in x],foutputs)))
    #outputs = np.round(np.array(list(map(lambda x : [min(1,i) for i in x],foutputs)))).astype(np.int8)
    outputs = np.round(np.array(list(map(lambda x : [min(1,i+0.25) for i in x],foutputs)))).astype(np.int8)
    segmap = SegmentationMapsOnImage(outputs, shape=outputs.shape)
    image = plt.imread(os.path.join(os.getcwd(),'mosaic_test.jpg'))
    x,y = (index)//5  , (index+1)%5
    if(y==0):   
        y=5
    img_crop=image[(x)*750:(x+1)*750,(y-1)*750:(y)*750,:]
    ans[(x)*750:(x+1)*750,(y-1)*750:(y)*750] = segmap.get_arr()
    image = img_crop
    side_by_side = np.hstack([
        segmap.draw_on_image(image),
        image.reshape((1,750,750,3)),
        segmap.draw()
    ]).reshape((750*3,750,3))
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.axis('off')
    plt.title('Segmentation masks for '+str(index+1))
    ax.imshow(side_by_side)
    savePATH = os.path.join(os.getcwd(),loc,str(index+1)+'.png')
    plt.savefig(savePATH,bbox_inches='tight')
    index+=1

np.save(os.path.join(os.getcwd(),loc,'out_imgds.npy'),ans)


