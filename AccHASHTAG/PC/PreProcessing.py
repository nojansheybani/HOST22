import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
import torchvision.models as models
from torch.nn import Module
from torch import nn
import copy
from hashtagcomm import *
from pearson_hash import *
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Model(Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)
        self.relu5 = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        y = self.relu4(y)
        y = self.fc3(y)
        y = self.relu5(y)
        return y

class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1)
        self.average1 = nn.AvgPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.average2 = nn.AvgPool2d(2, stride=2)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=4, stride=1)
        
        self.flatten = Flatten()
        
        self.fc1 = nn.Linear(120, 82)
        self.fc2 = nn.Linear(82,10)

    def forward(self, xb):
        xb = xb.view(-1, 1, 28, 28)
        xb = F.tanh(self.conv1(xb))
        xb = self.average1(xb)
        xb = F.tanh(self.conv2(xb))
        xb = self.average2(xb)
        xb = F.tanh(self.conv3(xb))
        xb = xb.view(-1, xb.shape[1])
        xb = F.relu(self.fc1(xb))
        xb = F.relu(self.fc2(xb))
        return xb

def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di

def preprocess():
    lenet5 = LeNet5()
#     torch.save(lenet5.state_dict(), "lenet.pth")
    lenet5.load_state_dict(torch.load("lenet.pth"))
    hash_dict = {}
    layerDict = {}
    i=0
    for name,_ in list(lenet5.named_children()):
        layerDict[i] = name
        i += 1
    i=0
    for layer in lenet5.children():
        layername = layerDict[i]
        i += 1
        if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
            file = open(f"./weights/{layername}.dat","w")
            weightlist = list(np.array(layer.state_dict()['weight']).flatten())
            for weight in weightlist:
                file.write(f"{weight}\n")
            file.close()

    for layer in layerDict.values():
        if "conv" in layer or "fc" in layer:
            hash_dict[layer] = main(layer)
    
    save_dict(hash_dict, "hashes.pkl")
    
    return lenet5, layerDict, hash_dict

def bitFlip(model, layer):
    noisy_dict = model.state_dict()
    weights = noisy_dict[f"{layer}.weight"]
    shape = weights.shape
    bitflip = np.ones(shape)
    print(shape)
    if bitflip.ndim == 4:
        bitflip[0][0][0][0] = -1
    else:
        bitflip[0][0] = -1
    noisy_dict[f"{layer}.weight"] = weights * bitflip
    model.load_state_dict(noisy_dict)
    weightlist = list(np.array(noisy_dict[f"{layer}.weight"]).flatten())
    file = open(f"./weights/{layer}.dat","w")
    for weight in weightlist:
        file.write(f"{weight}\n")
    file.close()        
    return model

def compare(pkl1, pkl2):
    og = load_dict(pkl1)
    flipped = load_dict(pkl2)
    for layer in og.keys():
        if flipped[layer] != og[layer]:
            print(f"BITFLIP FOUND IN {layer}")