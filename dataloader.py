from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import ImageDataset


def dataloader(train_dir, test_dir, crop_size, batch_size):
    input_transforms = transforms.Compose([
        transforms.Resize((crop_size, crop_size)),
        transforms.ToTensor(),
    ])

    train_dataset = ImageDataset(train_dir, input_transforms)
    test_dataset = ImageDataset(test_dir, input_transforms)

    training_data_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size)
    testing_data_loader = DataLoader(
        dataset=test_dataset, batch_size=batch_size)

    return training_data_loader, testing_data_loader
