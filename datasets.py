import os
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split, DataLoader
import torchvision.transforms as tt

data_dir = '/Users/salman/Downloads/Stat_learning_project/plant-seedlings-classification'  # Replace this with the path to your data folder

# Data transforms (normalization & data augmentation)
stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
train_tfms = tt.Compose([tt.RandomCrop(32, padding=4, padding_mode='reflect'), 
                         tt.RandomHorizontalFlip(), 
                         tt.ToTensor(), 
                         tt.Normalize(*stats, inplace=True)])
valid_tfms = tt.Compose([tt.ToTensor(), tt.Normalize(*stats)])


def get_data_loaders(batch_size=32, validation_split_ratio=0.1):
    # PyTorch datasets
    train_ds = ImageFolder(data_dir+'/train', train_tfms)

    # Split the dataset into train and validation datasets
    n_train = len(train_ds)
    n_validation = int(n_train * validation_split_ratio)
    n_train = n_train - n_validation
    train_ds, valid_ds = random_split(train_ds, [n_train, n_validation])

    # PyTorch data loaders
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=2, pin_memory=True)
    valid_dl = DataLoader(valid_ds, batch_size*2, num_workers=2, pin_memory=True)

    classes = os.listdir(data_dir + "/train")

    return train_dl, valid_dl, classes

#if __name__ == "__main__":
#    print(os.listdir(data_dir))
#    classes = os.listdir(data_dir + "/train")
#    print(classes)
