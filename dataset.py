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


class ImageDataset(Dataset):
    def __init__(self, image_dir, input_transforms=None):
        super(ImageDataset, self).__init__()
        self.image_filenames = [os.path.join(image_dir, x) for x in os.listdir(image_dir) if is_image_file(x)]
        self.input_transforms = input_transforms

    def __getitem__(self, index):
        input = Image.open(self.image_filenames[index])

        if self.input_transforms:
            input = self.input_transforms(input)
        
        return input
    
    def __len__(self):
        return len(self.image_filenames)