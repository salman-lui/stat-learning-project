import os
import torch
from torch.utils.data import DataLoader
from datasets import get_data_loaders
from utils import get_default_device, to_device, DeviceDataLoader, fit_one_cycle, evaluate, get_lr
from model import ResNet, ResNetPretrained
from plot import plot_accuracies, plot_losses, plot_lrs
from test_submission import generate_predictions


def main():
    train_dl, valid_dl, classes = get_data_loaders(batch_size=32, validation_split_ratio=0.1)

    # Set up the model
    num_classes = len(classes)
    model = ResNet(3, num_classes)

    # Move the model to the device
    device = get_default_device()
    model = to_device(model, device)

    # Set up the data loaders
    train_loader = DeviceDataLoader(train_dl, device)
    valid_loader = DeviceDataLoader(valid_dl, device)

    # Train the model
    epochs = 10
    max_lr = 0.01
    grad_clip = 0.1
    weight_decay = 1e-4
    history = fit_one_cycle(epochs, max_lr, model, train_loader, valid_loader, weight_decay, grad_clip)

    print("Training completed.")

    # Save the plots
    save_path = '/Users/salman/Downloads/Stat_learning_project/base_model/results'
    os.makedirs(save_path, exist_ok=True)

    plot_accuracies(history, save_path)
    plot_losses(history, save_path)
    plot_lrs(history, save_path)

    # Generate predictions and save them in a CSV file
    test_data_dir = "/Users/salman/Downloads/Stat_learning_project/plant-seedlings-classification/test"
    generate_predictions(model, test_data_dir, classes, device)



if __name__ == "__main__":
    main()
