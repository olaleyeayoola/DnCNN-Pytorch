from torch.utils.data import Dataset
from PIL import Image
import os
from utils import is_image_file


class ImageDataset(Dataset):
    def __init__(self, image_dir, input_transforms=None):
        super(ImageDataset, self).__init__()
        self.image_filenames = [os.path.join(
            image_dir, x) for x in os.listdir(image_dir) if is_image_file(x)]
        self.input_transforms = input_transforms

    def __getitem__(self, index):
        input = Image.open(self.image_filenames[index])

        if self.input_transforms:
            input = self.input_transforms(input)

        return input

    def __len__(self):
        return len(self.image_filenames)
