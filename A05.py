import torch
from torch import nn 
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split

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

            # Additional convolutional layer
            nn.Conv2d(64, 128, 3, padding="same"),
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
            # Data augmentation for training
            data_transform = v2.Compose([
                v2.ToImageTensor(), 
                v2.ConvertImageDtype()]) 
        else:
            # Data augmentation for test
            data_transform = v2.Compose([
                v2.ToImageTensor(), 
                v2.ConvertImageDtype()]) 
        return data_transform
    
    if approach_name == "CNN1":
        if training:
            # Data augmentation for training
            data_transform = v2.Compose([
                v2.ToImageTensor(), 
                v2.ConvertImageDtype()]) 
        else:
            # Data augmentation for test
            data_transform = v2.Compose([
                v2.ToImageTensor(), 
                v2.ConvertImageDtype()]) 
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
    # Training the model based on approach
    if approach_name == "CNN0":

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
        total_epochs = 20

        for epoch in range(total_epochs):
            print("** EPOCH", (epoch+1), "**")
            train(model, loss_fn, optimizer, device, train_dataloader)
            test(model, loss_fn, device, train_dataloader, "TRAIN")
            test(model, loss_fn, device, test_dataloader, "TEST")
        
    if approach_name == "CNN1": 
        
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
        total_epochs = 20

        for epoch in range(total_epochs):
            print("** EPOCH", (epoch+1), "**")
            train(model, loss_fn, optimizer, device, train_dataloader)
            test(model, loss_fn, device, train_dataloader, "TRAIN")
            test(model, loss_fn, device, test_dataloader, "TEST")

    return model
            
def train(model, loss_fn, optimizer, device, train_dataloader):
    # Train the model using train dataloader and return the trained model
    size = len(train_dataloader)
    model.train()
    
    for batch, (X,y) in enumerate(train_dataloader):
        X = X.to(device)
        y = y.to(device)
        
        pred = model(X)
        loss = loss_fn(pred, y)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if batch%100 == 0:
            loss = loss.item()
            index = (batch+1)*len(X)
            print(index, "of", size, ": Loss =", loss)

def test(model, loss_fn, device, test_dataloader, dataname):
    # Test the model using test dataloader and return the trained model
    size = len(test_dataloader)
    num_batches = len(test_dataloader)
    model.eval()
    
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for X,y in test_dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
    
    print(dataname, ":")
    print("\tAccuracy:", correct)
    print("\tLoss:", test_loss)