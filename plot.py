import matplotlib.pyplot as plt
import numpy as np
import os

def plot_accuracies(history, save_path):
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs')
    plt.savefig(os.path.join(save_path, 'accuracy_plot.png'))
    plt.close()

def plot_losses(history, save_path):
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs')
    plt.savefig(os.path.join(save_path, 'loss_plot.png'))
    plt.close()

def plot_lrs(history, save_path):
    lrs = np.concatenate([x.get('lrs', []) for x in history])
    plt.plot(lrs)
    plt.xlabel('Batch no.')
    plt.ylabel('Learning rate')
    plt.title('Learning Rate vs. Batch no.')
    plt.savefig(os.path.join(save_path, 'learning_rate_plot.png'))
    plt.close()
