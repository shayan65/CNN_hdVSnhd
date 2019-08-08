import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import cv2
import os
import keras
from torch import optim #function for the optimization gradient descent
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
from torchvision import transforms, datasets, models

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data
import torchvision

'''
In this data challenge I'll show how to start from pretrained neural network (transfer learning) with Pytorch to train our hotdog classification model and then predict other images.
Here are the steps to train our CNN classifier and apply it to our system:
1- Oraginize the training and test sets
2- Image augmentation
3- Apply transfer learning to pre-train our image classifier (load the pre-trained model)
4- Validate our model
'''



'''
1- Oraginize the training and test sets:
Here we define the train / validation dataset loader, using the SubsetRandomSampler for the split:
------------------------------------------------------------------------------------------------------------------------
Load the data organize the training dataset
1- Training data
Directory for training images: train_dir = '../dataset/hd'
2- Validation data
Directory for validation images: valid_dir = '../dataset/hd'
3- Test Data
Directory for testing images: test_dir = '../dataset/nhd'
------------------------------------------------------------------------------------------------------------------------
'''
data_dir = '../dataset'
transform = transforms.Compose(
                               [transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car',



stop
current_path = os.getcwd()
datadir = "../dataset"
train_dir = os.path.join(current_path, "../dataset/hd/")
valid_dir =os.path.join(current_path, "../dataset/hd/")
test_dir = os.path.join(current_path, "../dataset/nhd/")
print(train_dir)


'''
------------------------------------------------------------------------------------------------------------------------
Categories:
There are two classes in this problem, hot dogs (hd) and none hot dogs (nhd)
------------------------------------------------------------------------------------------------------------------------
'''
CATEGORIES = ["hd", "nhd"]



'''
2- Image augumentation
------------------------------------------------------------------------------------------------------------------------
Transfromation for the training: image augumention
------------------------------------------------------------------------------------------------------------------------
'''
EPOCHS = 2
BATCH_SIZE = 10
                       
                       


# Define transforms for the training, validation, and testing sets
training_transforms = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

validation_transforms = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                                 [0.229, 0.224, 0.225])])

testing_transforms = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])

# TODO: Load the datasets with ImageFolder
training_dataset = datasets.ImageFolder(train_dir, transform=training_transforms)
validation_dataset = datasets.ImageFolder(valid_dir, transform=validation_transforms)
testing_dataset = datasets.ImageFolder(test_dir, transform=testing_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=64, shuffle=True)
validate_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=32)
test_loader = torch.utils.data.DataLoader(testing_dataset, batch_size=32)
                                                                       
                                                                       
                                                                       

# Build and train your network
# Transfer Learning
model = models.vgg16(pretrained=True)
model

# Freeze pretrained model parameters to avoid backpropogating through them
for parameter in model.parameters():
    parameter.requires_grad = False


from collections import OrderedDict

# Build custom classifier
classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(25088, 5000)),
                                        ('relu', nn.ReLU()),
                                        ('drop', nn.Dropout(p=0.5)),
                                        ('fc2', nn.Linear(5000, 102)),
                                        ('output', nn.LogSoftmax(dim=1))]))

model.classifier = classifier
