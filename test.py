import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
from torch.autograd import Variable

device = 'cuda'
# model = SRNET().to(device)
model = torch.load('/content/model_epoch_175.pth')
model = model['arch']
loader = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor()
    ])


def image_loader(image_name):
    """load image, returns tensor"""
    image = Image.open(image_name)
    image = loader(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0) 
    return image.cuda()


def predict(classifer, image):
    out = classifer(image)
    out = out.cpu().clone()
    out = out.squeeze(0)
    trans = transforms.ToPILImage()
    plt.imshow(trans(out))
    trans(out).save('out.png')


img_path = '/content/drive/My Drive/SR_Data/test/lr/5_09.png'
# pass the image into the image_loader function
image = image_loader(img_path)
# get prediction
predict(model, image)