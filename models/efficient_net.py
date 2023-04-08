import torch
import timm
import torch.nn as nn
import torch.nn.functional as F

class EfficientNetTimm(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Load the pre-trained model from timm
        self.model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
        # Add a new classifier for your dataset
        self.classifier = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = self.classifier(x)
        return x

    def training_step(self, batch):
        inputs, targets = batch
        outputs = self(inputs)  # Forward pass
        loss = F.cross_entropy(outputs, targets)  # Compute loss
        return loss

    def validation_step(self, batch):
        inputs, targets = batch
        outputs = self(inputs)
        loss = F.cross_entropy(outputs, targets)
        _, preds = torch.max(outputs, 1)
        acc = torch.tensor(torch.sum(preds == targets).item() / len(preds))
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))
