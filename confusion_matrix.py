# BELOW CODE IS GENRATING CONFUSION-MATRIX

import os
import numpy as np
import torch
import pickle
from torchvision import transforms
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataloader import custom_dset  # Your custom dataset module
from model import embedding  # Your model module
from torch.utils.data import DataLoader
from dataloader.triplet_img_loader import BaseLoader

def load_model(checkpoint_path, device):
    embeddingNet = embedding.EmbeddingResnet()  # Adjust as needed
    model_dict = None
    if os.path.isfile(checkpoint_path):
        print(f"=> Loading checkpoint '{checkpoint_path}'")
        try:
            model_dict = torch.load(checkpoint_path, weights_only=True)['state_dict']
        except Exception:
            model_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)['state_dict']
        print(f"=> Loaded checkpoint '{checkpoint_path}'")
    else:
        print(f"=> No checkpoint found at '{checkpoint_path}'")
        return None

    model_dict_mod = {}
    for key, value in model_dict.items():
        new_key = '.'.join(key.split('.')[2:])
        model_dict_mod[new_key] = value
    model = embeddingNet.to(device)
    model.load_state_dict(model_dict_mod)
    return model

def generate_predictions(data_loader, model, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in tqdm(data_loader):
            batch_imgs1, batch_imgs2, batch_imgs3 = data  # Assuming this returns triplets
            images = torch.cat((batch_imgs1, batch_imgs2, batch_imgs3), dim=0).to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend([0] * batch_imgs1.size(0))
            all_labels.extend([1] * batch_imgs2.size(0))
            all_labels.extend([2] * batch_imgs3.size(0))

    return np.array(all_labels), np.array(all_preds)

def compute_confusion_matrix(all_labels, all_preds):
    return confusion_matrix(all_labels, all_preds)

def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()

def main():
    torch.manual_seed(1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint_path = 'C://Users//accurate//Desktop//Dhananjay//Siamese-Network-Using-Triplet-Loader-For-TLess-Dataset//pytorch-siamese-triplet//results//Custom_exp1//checkpoint_100.pth'
    # base_path = 'C://Users//accurate//Desktop//Dhananjay//Siamese-Network-Using-Triplet-Loader-For-TLess-Dataset//custom'
    base_path = 'C://Users//accurate//Desktop//Dhananjay//Siamese-Network-Using-Triplet-Loader-For-TLess-Dataset//custom'

    # Load the trained model
    model = load_model(checkpoint_path, device)

    # Prepare your custom dataset
    dset_obj = custom_dset.Custom()  # Instantiate your custom dataset
    dset_obj.load(base_path)  # Load dataset and generate triplets
    test_triplets = [dset_obj.getTriplet() for _ in range(100)]  # Update with the number of test samples

    transform = transforms.Compose([
        transforms.ToTensor(),  # Customize as needed
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_loader = DataLoader(BaseLoader(test_triplets, transform=transform), batch_size=32, shuffle=False)

    # Generate predictions
    all_labels, all_preds = generate_predictions(test_loader, model, device)

    # Compute confusion matrix
    cm = compute_confusion_matrix(all_labels, all_preds)

    # Define your class names
    class_names = ['class_1', 'class_3','class_5','class_7','class_8','class_9','class_10']  # Update with actual class names

    # Plot the confusion matrix
    plot_confusion_matrix(cm, class_names)

if __name__ == '__main__':
    main()
