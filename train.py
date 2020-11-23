import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from utils import gaussian, save_checkpoint
from model import DnCNN
from dataloader import dataloader
import argparse
from math import log10


def train(epoch, model, optimizer, training_data_loader, mean,
          stddev, criterion):
    epoch_loss = 0
    for iteration, data in enumerate(training_data_loader, 1):

        target = data
        _input, noise = gaussian(target, mean, stddev)

        if device == 'cuda':
            _input = Variable(_input.cuda())
            noise = Variable(noise.cuda())
            target = Variable(target.cuda())

        output = model(_input)

        loss = criterion(output, noise)
        epoch_loss += loss.item()

        loss.backward()

        optimizer.step()

        optimizer.zero_grad()

        print("===> Epoch[{}]({}/{}): Loss: {:.4f}"
              .format(epoch,
                      iteration,
                      len(training_data_loader),
                      loss.item()))

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(
        epoch, epoch_loss / len(training_data_loader)))


def validate(model, testing_data_loader, mean, stddev, criterion):
    avg_psnr = 0
    model.eval()
    with torch.no_grad():
        for data in testing_data_loader:
            target = data
            _input, noise = gaussian(target, mean, stddev)

            _input = Variable(_input.cuda())
            noise = Variable(noise.cuda())
            target = Variable(target.cuda())

            output = model(_input)
            mse = criterion(output, noise)
            psnr = 10 * log10(1 / mse.item())
            avg_psnr += psnr
    print("===> Avg. PSNR: {:.4f} dB".format(
        avg_psnr / len(testing_data_loader)))


def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)

    # DATA PATHS
    parser.add_argument("--train_dir", default='data/BSDS300/images/train',
                        help="train data directory")
    parser.add_argument("--test_dir", default='data/BSDS300/images/test',
                        help="test data directory")
    parser.add_argument("--crop_size", default=128,
                        type=int, help="size to resize images")
    parser.add_argument("--batch_size", default=128,
                        type=int, help="train batch size")

    # NOISE
    parser.add_argument("--noise_mean", default=0.0,
                        type=float, help="gausian noise mean")
    parser.add_argument("--noise_std", default=0.05, type=float,
                        help="gausian noise standard deviation")

    # TRAINING DATA
    parser.add_argument("--lr", default=1e-3, type=float, help="learning rate")
    parser.add_argument("--num_epochs", default=1,
                        type=int, help="number of epochs")
    parser.add_argument("--eval_interval", default=1,
                        type=int, help="evaluation interval")
    parser.add_argument("--save_interval",
                        default=1,
                        type=int,
                        help="number of epochs")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DnCNN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    training_data_loader, testing_data_loader = dataloader(
        args.train_dir, args.test_dir, args.crop_size, args.batch_size)

    mean = args.noise_mean
    stddev = args.noise_std
    num_epochs = args.num_epochs
    for epoch in range(1, num_epochs + 1):
        train(epoch, model, optimizer,
              training_data_loader, mean, stddev, criterion)

        if epoch % args.eval_interval == 0:
            validate(model, testing_data_loader, mean, stddev, criterion)

        if epoch % args.save_interval == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': model,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            })
