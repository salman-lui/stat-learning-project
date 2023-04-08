import os
import torch
import argparse
from torch.utils.data import DataLoader
from datasets import get_data_loaders
from utils import get_default_device, to_device, DeviceDataLoader, fit_one_cycle, evaluate, get_lr
from plot import plot_accuracies, plot_losses, plot_lrs
from test_submission import generate_predictions


from models.resnet import ResNet, ResNetTimm
from models.efficient_net import EfficientNetTimm


# script 
#python main.py --save_path /path/to/save/results --epochs 10 --model ResNetPretrained


def main(args):
    train_dl, valid_dl, classes = get_data_loaders(batch_size=32, validation_split_ratio=0.1)

    # Set up the model
    num_classes = len(classes)

    if args.model == "ResNet":
        model = ResNet(3, num_classes)
    elif args.model == "ResNetPretrained":
        model = ResNetTimm(num_classes)
    elif args.model == "EfficientNetPretrained":
        model = EfficientNetTimm(num_classes)
    else:
        raise ValueError(f"Invalid model name: {args.model}")

    # Move the model to the device
    device = get_default_device()
    model = to_device(model, device)

    # Set up the data loaders
    train_loader = DeviceDataLoader(train_dl, device)
    valid_loader = DeviceDataLoader(valid_dl, device)

    # Train the model
    epochs = args.epochs
    max_lr = 0.01
    grad_clip = 0.1
    weight_decay = 1e-4
    history = fit_one_cycle(epochs, max_lr, model, train_loader, valid_loader, weight_decay, grad_clip)

    print("Training completed.")

    # Save the plots
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)

    plot_accuracies(history, save_path)
    plot_losses(history, save_path)
    plot_lrs(history, save_path)

    # Generate predictions and save them in a CSV file
    test_data_dir = "/Users/salman/Downloads/Stat_learning_project/plant-seedlings-classification/test"
    generate_predictions(model, test_data_dir, classes, device, save_path) #save_path where you want to store the data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Deep Learning model")
    parser.add_argument("--save_path", type=str, default="/Users/salman/Documents/GitHub/stat-learning-project/results", help="Path to save the results")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training")
    parser.add_argument("--model", type=str, default="ResNet", help="Choose model ResNet and ResNetPretrained, EfficientNetPretrained etc .... ")
    args = parser.parse_args()
    main(args)

