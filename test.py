import torch
import torchvision.transforms as transforms
from model import DnCNN
from utils import image_loader, predict
import argparse


def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)

    # DATA PATHS
    parser.add_argument("--image_path", default='images/guassian.png',
                        help="path to image")
    parser.add_argument("--save_image_path", default='images',
                        help="path to save image to")
    parser.add_argument("--crop_size", default=224, type=int,
                        help="size to resize image to")
    parser.add_argument("--model_path",
                        default='trained_model/model_epoch_10.pth',
                        help="path to saved model")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DnCNN().to(device)
    model = torch.load(args.model_path,  map_location=device)
    model = model['arch']

    loader = transforms.Compose([
        transforms.Resize((args.crop_size, args.crop_size)),
        transforms.ToTensor()
    ])

    img_path = args.image_path

    # pass the image into the image_loader function
    image = image_loader(img_path, loader, device)

    # get prediction
    predict(model, image, args.save_image_path)
