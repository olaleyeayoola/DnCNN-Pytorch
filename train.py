import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from matplotlib import pyplot as plt
import os
import cv2
import numpy as np
from math import log10
from torch.autograd import Variable
from skimage.util import random_noise
from google.colab.patches import cv2_imshow


def train(epoch):
    epoch_loss = 0
    for iteration, data in enumerate(training_data_loader, 1):
        model.train()
        optimizer.zero_grad()

        target = data
        input = gaussian(target, True, 0, 0.05)
        
        input = Variable(input.cuda())
        target = Variable(target.cuda())
        
        output = model(input)

        loss = criterion(output, target)
        epoch_loss += loss.item()
        
        loss.backward()
        optimizer.step()

        print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(training_data_loader), loss.item()))

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))

def validate():
    avg_psnr = 0
    model.eval()
    with torch.no_grad():
        for data in testing_data_loader:
            target = data
            input = gaussian(target, True, 0, 0.05)

            input = Variable(input.cuda())
            target = Variable(target.cuda())

            output = model(input)
            mse = criterion(output, target)
            psnr = 10 * log10(1 / mse.item())
            avg_psnr += psnr
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))