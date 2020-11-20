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
from torch.autograd import Variable
import utils
from .model import DnCNN
import argparse
from skimage.measure.simple_metrics import compare_psnr


def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i, :, :, :],
                             Img[i, :, :, :], data_range=data_range)
    return (PSNR/Img.shape[0])


def save_checkpoint(state):
    model_out_path = "model_epoch_{}.pth".format(epoch)
    torch.save(state, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


def train(epoch, model, optimizer, training_data_loader, gaussian, criterion):
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

        print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(
            epoch,
            iteration,
            len(training_data_loader),
            loss.item()
        ))

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(
        epoch, epoch_loss / len(training_data_loader)))


def validate(model, testing_data_loader, gaussian, criterion):
    avg_psnr = 0
    model.eval()
    with torch.no_grad():
        for data in testing_data_loader:
            target = data
            input, noise = gaussian(target, True, 0, 0.05)

            input = Variable(input.cuda())
            noise = Variable(noise.cuda())
            target = Variable(target.cuda())

            output = model(input)
            # mse = criterion(output, noise)
            psnr = batch_PSNR(output, input, 1.)
            avg_psnr += psnr
    print("===> Avg. PSNR: {:.4f} dB".format(
        avg_psnr / len(testing_data_loader)))


def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)

    # DATA PATHS
    parser.add_argument("--train_path", default='data',
                        help="train data directory")
    parser.add_argument("--test_path", default='data',
                        help="test data directory")
    parser.add_argument("--batch_size", default=128,
                        type=int, help="train batch size")

    # NOISE
    parser.add_argument("--noise_mean", default=0,
                        type=int, help="gausian noise mean")
    parser.add_argument("--noise_std", default=0, type=int,
                        help="gausian noise standard deviation")

    # TRAINING DATA
    parser.add_argument("--lr", default=1e-3, type=float, help="learning rate")
    parser.add_argument("--num_epochs", default=100,
                        type=int, help="number of epochs")
    parser.add_argument("--eval_interval", default=25,
                        type=int, help="evaluation interval")
    parser.add_argument("--save_interval",
                        default=25,
                        type=int,
                        help="number of epochs")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 10
    for epoch in range(1, num_epochs + 1):
        train(epoch)
        if epoch % 1 == 0:
            validate()
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': model,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            })
