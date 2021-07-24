#!/usr/bin/env python
# coding: utf-8

# Importing Libraries:
# ==

# In[1]:



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
    #torch.set_default_tensor_type('torch.cuda.FloatTensor') (else HPC doesn't work)
else:
    print('GPU is not available!')
    device = "cpu"


# Hyperparams:
# ==

# In[2]:


# depth of UNet = 4
learning_rate = 1e-4
epoch_number = 200
batch_size = 64 
train_size = 100 # divide train:val by 70:30 probably
aug_train_size = 500
test_size = 25 
img_size = 256 # dimensions to resize input 750x750 into (power of 2^(depth+\delta) necessary for model to work)
min_channels = 16 
gradient_clip = 1e9 # basically unactive
dropout_percent = 35 
dice_alpha = 1e-3 # maybe try 1e-4 and 1.0
pretrained = True
pretrained_model = 'unet'
freeze_layers = 1 # for pretrained model
name = 'j_test200'
PATH = os.path.join(os.getcwd(),'models/',name+'.pt')
bestPATH = os.path.join(os.getcwd(),'models/best_model_'+name+'.pt')


# Loading Data:
# ==

# In[3]:


def transform(x):
    return  torchvision.transforms.functional.to_tensor(x)

def augmenter(image, segmap, index):
    mode = index//(train_size)
    if(mode == 1):
        image, segmap = iaa.Sequential([
            iaa.CoarseDropout(0.1, size_percent=0.2),
            iaa.Affine(rotate=(-30, 30)),
            iaa.ElasticTransformation(alpha=20, sigma=2)
        ])(image=image, segmentation_maps=segmap)
    elif(mode == 2):
        image, segmap = iaa.Sequential([
            iaa.Dropout([0.05, 0.2]),      
            iaa.Sharpen((0.0, 1.0)),       
            iaa.Affine(rotate=(-90, 90)),  
            iaa.ElasticTransformation(alpha=50, sigma=5)  
        ], random_order=True)(image=image, segmentation_maps=segmap)
    elif(mode == 3):
        image, segmap = iaa.Sequential([
            iaa.Fliplr(p=0.8),
            iaa.Flipud(p=0.8),
            iaa.Affine(rotate = (-180,180))
        ])(image=image, segmentation_maps=segmap)
    elif(mode == 4):
        image, segmap = iaa.Sequential([
            iaa.Fliplr(p=1),
            iaa.Flipud(p=0.4),
            iaa.Affine(scale=(0.5,1.5),rotate=(-120,120)),
            iaa.WithHueAndSaturation([
                iaa.WithChannels(0, iaa.Add((-30, 10))),
                iaa.WithChannels(1, [
                    iaa.Multiply((0.5, 1.5)),
                    iaa.LinearContrast((0.75, 1.25))
                ])
            ])
        ])(image=image, segmentation_maps=segmap)
    return image, segmap
    

class DataSet(torch.utils.data.Dataset):
    def float_maker(self, a):
        if a[0]=='(':
            return a[1:]
        if a[-1]==')':
            return a[:-1]
        return a
    def __init__(self, mode):
        super().__init__()
        self.image_names= os.listdir('./Data_ML/image_chips/')
        self.mode = mode
        # self.image_names= os.listdir('/content/drive/MyDrive/Data_ML/image_chips/') for google.colab
    def __getitem__(self, index):
        if(self.mode == 'Train'):
            image = plt.imread(os.path.join(os.getcwd(),'Data_ML','image_chips',self.image_names[index%train_size]))
            mask = pd.read_csv(os.path.join(os.getcwd(),'Data_ML','target_feature_AOI',self.image_names[index%train_size][:-4]+'.csv'))
            # for google.colab
            # image = plt.imread(os.path.join('/content/drive/MyDrive','Data_ML','image_chips',self.image_names[index%train_size]))
            # mask = pd.read_csv(os.path.join('/content/drive/MyDrive','Data_ML','target_feature_AOI',self.image_names[index%train_size][:-4]+'.csv'))
            coordinates = mask.WKT.values[0][16:-3].split(',')
            points = ([tuple(self.float_maker(i)  for i in x.split()) for x in coordinates])
            h , w = 750, 750
            point_path = Path(points)
            x, y = np.mgrid[:h, :w]
            # TODO: find if first or second method is correct, so far method 2 seems more correct
            #y = - y
            #coors=np.hstack((x.reshape(-1, 1), y.reshape(-1,1)))
            x = -x
            coors=np.array(list(map(lambda a : (a[1],a[0]),(np.hstack((x.reshape(-1, 1), y.reshape(-1,1)))))))
            masked_image = (point_path.contains_points(coors)).reshape(h,w)
            segmap = SegmentationMapsOnImage(masked_image, shape = masked_image.shape)
            image, segmap_aug = augmenter(image, segmap, index)
            masked_image = segmap_aug.get_arr()
            first = torchvision.transforms.Resize((img_size,img_size))(transform(image.copy()))
            second = (torchvision.transforms.functional.to_tensor(masked_image).int())
            return (first,second)
        else:
            image = plt.imread(os.path.join(os.getcwd(),'mosaic_test.jpg'))
            x,y = (index)//5  , (index+1)%5
            if(y==0):   
                y=5
            img_crop=image[(x)*750:(x+1)*750,(y-1)*750:(y)*750,:]
            return torchvision.transforms.Resize((img_size,img_size))(transform(img_crop))
    def __len__(self):
        if(self.mode=='Train'):
            return aug_train_size
        else:
            return test_size


# In[4]:


# input training images are 750x750x3 and are 100 in number
# test image is 5x5 images (hence 3750x3750x3), we need to output a masked image with 0s and 1s
input_channels = 3 
output_classes = 1

train_dataLoader = data.DataLoader(
    DataSet('Train'), #need to get DataSet to make train and CV both
    batch_size = batch_size,
    shuffle = True,
    num_workers= 0
)
test_dataLoader = data.DataLoader(
    DataSet('Test'),
    batch_size=1,
    shuffle=False,
    num_workers=0
)


# Creating Module:
# ==

# In[5]:


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
    Model = Model.train(True).to(device)    
    for parents in Model.children():
        for count, child in enumerate(parents.children()):
            if count == freeze_layers:
                break
            for param in child.parameters():
                param.requires_grad=False
else:
    Model = UNet(input_channels, min_channels)
    Model = Model.to(device)

if(use_gpu):
   Model = nn.DataParallel(Model, device_ids = [i for i in range(torch.cuda.device_count())]).to(0)


optimizer = optim.Adam(Model.parameters(),lr=learning_rate)


# In[6]:


def save_ckp(state, is_best, checkpoint_path, best_model_path):
    f_path = checkpoint_path
    torch.save(state, f_path)
    if is_best:
        best_fpath = best_model_path
        shutil.copyfile(f_path, best_fpath)
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


# Training Model:
# ==

# In[ ]:


# maybe make into a functional model 'def train' so that i can use checkpoints as well
runningLoss_min = 1e9 # take input for this from previous training if using functional model and training multiple times
epochs = epoch_number

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = dice_alpha #maybe try 1e-4
    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (2. * intersection + self.smooth) / (y_pred.sum() + y_true.sum() + self.smooth)
        return 1. - dsc

loss_func = DiceLoss()
acc_sanity = jsc
trainLoss = []
sanityAccc = []
liveloss = PlotLosses()
for epoch in range(epochs):  
    epochStart = time.time()
    runningLoss = 0.0
    sanityAcc = 0.0
    i = 0
    for inputs, labels in tqdm(train_dataLoader): 
        if(i==3):
            break
        inputs = inputs.to(device)
        if(use_gpu):
            inputs = inputs.to(0)
        labels = labels.to(device)
        if(pretrained):
            inputs = torchvision.transforms.Resize((256,256))(inputs)
        optimizer.zero_grad()  
        outputs = Model(inputs) 
        outputs = torchvision.transforms.Resize((750,750))(outputs)
        loss = loss_func(outputs, labels)
        loss.backward() 
        acc_san = acc_sanity(torch.round(outputs).cpu().detach().numpy().reshape(-1), labels.cpu().detach().numpy().reshape(-1))
        nn.utils.clip_grad_norm_(Model.parameters(), gradient_clip)
        optimizer.step()
        runningLoss += loss.item()
        sanityAcc += acc_san.item()
    runningLoss *= batch_size/aug_train_size
    sanityAcc *= batch_size/aug_train_size
    logs = {}
    logs['loss'] = runningLoss
    logs['accuracy'] = sanityAcc
    liveloss.update(logs)
    liveloss.send()
    checkpoint = {
            'epoch': epoch + 1,
            'loss': runningLoss,
            'state_dict': Model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'accuracy' : sanityAcc,
        }
    save_ckp(checkpoint, False, PATH, bestPATH)
    # do the below for validation loss 
    if runningLoss <= runningLoss_min:
        print('Loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(runningLoss_min,runningLoss))
        save_ckp(checkpoint, True, PATH, bestPATH)
        runningLoss_min = runningLoss
    sanityAccc.append(sanityAcc)
    trainLoss.append(runningLoss)
    epochEnd = time.time()-epochStart
    print('Iteration: {:.0f} /{:.0f}  ;  Training Loss: {:.6f} ; Jaccard Score: {:.6f} ; Time consumed: {:.0f}m {:.0f}s '        .format(epoch + 1, epochs, runningLoss, sanityAcc, epochEnd//60, epochEnd%60))   
print('Finished Training')







