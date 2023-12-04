import torch
from torch import nn 
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import cv2
import numpy as np
import os

class CNN0_Network(nn.Module):
    def __init__(self, class_cnt):
        super().__init__()        
        self.net_stack = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(32,32, 3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Flatten(),
            
            nn.Linear(4096, 32),
            nn.ReLU(),
            nn.Linear(32, class_cnt)
        )

    def forward(self, x):        
        logits = self.net_stack(x)
        return logits

class CNN1_Network(nn.Module):
    def __init__(self, class_cnt):
        super().__init__()        
        self.net_stack = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(32,32, 3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding="same"),  # Additional convolutional layer
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Flatten(),
            
            nn.Linear(128 * 4 * 4, 32),  # Adjust the input size for the additional layer
            nn.ReLU(),
            nn.Linear(32, class_cnt)
        )
        
    def forward(self, x):        
        logits = self.net_stack(x)
        return logits

def get_approach_names():
    # Return a list of approach names
    return ["CNN0", "CNN1"] 

def get_approach_description(approach_name):
    # Return a description for the given approach_name
    if approach_name == "CNN0":
        return "Baseline CNN with standard architecture."
    elif approach_name == "CNN1":
        return "CNN with additional layers."
    else:
        return "Unknown approach."

def get_data_transform(approach_name, training):
    # Return the appropriate data transform based on approach_name and training flag
    if approach_name == "CNN0":
        if training:
            # Add data augmentation for training
            data_transform = v2.Compose([
                v2.RandomHorizontalFlip(),
                v2.RandomResizedCrop(32),
                v2.ToImageTensor(),
                v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        else:
            # For validation and testing, perform standard transformation
            data_transform = v2.Compose([
                v2.ToImageTensor(), 
                v2.ConvertDtype()]) 
        return data_transform
    
    if approach_name == "CNN1":
        if training:
            # Add data augmentation for training
            data_transform = v2.Compose([
                v2.RandomHorizontalFlip(),
                v2.RandomResizedCrop(32),
                v2.ToImageTensor(),
                v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        else:
            # For validation and testing, perform standard transformation
            data_transform = v2.Compose([
                v2.ToImageTensor(), 
                v2.ConvertDtype()]) 
        return data_transform

def get_batch_size(approach_name):
    # Return the preferred batch size for the given approach_name
    if approach_name == "CNN0":
        return 64
    elif approach_name == "CNN1":
        return 128
    else:
        return 32

def create_model(approach_name, class_cnt):
    # Create and return the PyTorch model based on approach_name
    if approach_name == "CNN0":
        return CNN0_Network(class_cnt)
    elif approach_name == "CNN1":
        return CNN1_Network(class_cnt)
    else:
        return None

def train_model(approach_name, model, device, train_dataloader, test_dataloader):

    if approach_name == "CNN0": 
        # Train the model using the provided dataloaders and return the trained model
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Train for a number of epochs (you can adjust this)
        total_epochs = 2
        
        for epoch in range(total_epochs):
            print("** EPOCH", (epoch+1), "**")
            train_model(model, loss_fn, optimizer, device, 
                            train_dataloader)

        # Optionally print and evaluate on test_dataloader
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print("Accuracy on test set: ", {accuracy})

        return model
    
    if approach_name == "CNN1": 
        # Train the model using the provided dataloaders and return the trained model
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Train for a number of epochs (you can adjust this)
        total_epochs = 20
        
        for epoch in range(total_epochs):
            print("** EPOCH", (epoch+1), "**")
            train_model(model, loss_fn, optimizer, device, 
                            train_dataloader)

        # Optionally print and evaluate on test_dataloader
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print("Accuracy on test set: ", {accuracy})

        return model