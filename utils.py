import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg",
                                                              ".jpeg", ".gif"])


def gaussian(ins, mean, stddev):
    noise = Variable(ins.data.new(ins.size()).normal_(mean, stddev))
    return ins + noise, noise


def save_checkpoint(state):
    model_out_path = f"model_epoch_{state.epoch}.pth"
    torch.save(state, model_out_path)
    print(f"Checkpoint saved to {model_out_path}")


def image_loader(image_name, loader, device):
    """load image, returns tensor"""
    image = Image.open(image_name)
    image = loader(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)
    if device == 'cuda':
        return image.cuda()
    else:
        return image.cpu()


def predict(classifer, image, save_path):
    out = classifer(image)
    out = out.cpu().clone()
    out = out.squeeze(0)
    trans = transforms.ToPILImage()
    # plt.imshow(trans(out))
    trans(out).save(f'{save_path}/out.png')
