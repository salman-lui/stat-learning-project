import os
import torch
from torchvision import transforms as T
from PIL import Image
import pandas as pd
from utils import get_default_device

imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

# Define your custom transform
custom_transforms = T.Compose([
    T.Resize((32, 32)),
    T.ToTensor(),
    T.Normalize(*imagenet_stats)
])

def load_preprocess_image(img_path, transforms):
    img = Image.open(img_path)
    img = img.convert('RGB')
    img = transforms(img)
    return img

def generate_predictions(model, test_data_dir, classes, device):
    results = []

    for file in os.listdir(test_data_dir):
        img_path = os.path.join(test_data_dir, file)
        img = load_preprocess_image(img_path, custom_transforms)
        img = img.unsqueeze(0).to(device)

        # Get prediction
        with torch.no_grad():
            output = model(img)
            _, preds = torch.max(output, dim=1)
            predicted_class = classes[preds.item()]

        results.append({"file": file, "species": predicted_class})

    # Save results to a CSV file
    results_df = pd.DataFrame(results)
    results_df.to_csv("predictions.csv", index=False)
