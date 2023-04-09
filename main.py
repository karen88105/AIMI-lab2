import copy
import os
import torch
import argparse
import dataloader
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from copy import deepcopy
import torch.optim as optim
import matplotlib.pyplot as plt
from models.EEGNet import EEGNet
from torchsummary import summary
from matplotlib.ticker import MaxNLocator
from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# from torchvision.transforms import Compose, RandomHorizontalFlip, RandomRotation

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
num = 33  #images name number

class BCIDataset(Dataset):
    def __init__(self, data, label):  #, transform=None
        self.data = data
        self.label = label
        # self.transform = transform

    def __getitem__(self, index):
        data = torch.tensor(self.data[index,...], dtype=torch.float32)
        label = torch.tensor(self.label[index], dtype=torch.int64)

        # if self.transform:
        #     data = self.transform(data)

        return data, label

    def __len__(self):
        return self.data.shape[0]

def plot_train_acc(train_acc_list, epochs):
    # TODO plot training accuracy
    num_epoch = []
    for i in range(epochs):
        num_epoch.append(i+1)
    plt.plot(num_epoch, train_acc_list)
    plt.title('Training Accuracy')
    plt.ylabel("Accuracy")
    plt.xlabel('Epoch')
    
    plt.legend(['train acc'], loc='upper left')
    
    plt.savefig('result_image/train_acc_' + str(num) + '.png')
    plt.show()


def plot_train_loss(train_loss_list, epochs):
    # TODO plot training loss
    num_epoch = []
    for i in range(epochs):
        num_epoch.append(i+1)
    plt.plot(num_epoch, train_loss_list)
    plt.title('Training Loss')
    plt.ylabel("Loss")
    plt.xlabel('Epoch')
    
    plt.legend(['train loss'], loc='upper right')
    
    plt.savefig('result_image/train_loss_' + str(num) + '.png')
    plt.show()

def plot_test_acc(test_acc_list, epochs):
    # TODO plot testing loss
    num_epoch = []
    for i in range(epochs):
        num_epoch.append(i+1)
    plt.plot(num_epoch, test_acc_list)
    plt.title('Testing Accuracy')
    plt.ylabel("Accuracy")
    plt.xlabel('Epoch')
    
    plt.legend(['test acc'], loc='upper left')
    
    plt.savefig('result_image/test_acc_' + str(num) + '.png')
    plt.show()

def train(model, loader, criterion, optimizer, args):
    best_acc = 0.0
    best_wts = None
    avg_acc_list = []
    test_acc_list = []
    avg_loss_list = []
    for epoch in range(1, args.num_epochs+1):
        model.train()
        with torch.set_grad_enabled(True):
            avg_acc = 0.0
            avg_loss = 0.0 
            for i, data in enumerate(tqdm(loader), 0):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                avg_loss += loss.item()
                _, pred = torch.max(outputs.data, 1)
                avg_acc += pred.eq(labels).cpu().sum().item()

            avg_loss /= len(loader.dataset)
            avg_loss_list.append(avg_loss)
            avg_acc = (avg_acc / len(loader.dataset)) * 100
            avg_acc_list.append(avg_acc)
            print(f'Epoch: {epoch}')
            print(f'Loss: {avg_loss}')
            print(f'Training Acc. (%): {avg_acc:3.2f}%')

        test_acc = test(model, test_loader)
        test_acc_list.append(test_acc)
        if test_acc > best_acc:
            best_acc = test_acc
            best_wts = model.state_dict()
        print(f'Test Acc. (%): {test_acc:3.2f}%')

    torch.save(best_wts, './weights/best.pt')
    return avg_acc_list, avg_loss_list, test_acc_list


def test(model, loader):
    avg_acc = 0.0
    model.eval()
    with torch.set_grad_enabled(False):
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, pred = torch.max(outputs, 1)
            for i in range(len(labels)):
                if int(pred[i]) == int(labels[i]):
                    avg_acc += 1

        avg_acc = (avg_acc / len(loader.dataset)) * 100

    return avg_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-num_epochs", type=int, default=500)
    parser.add_argument("-batch_size", type=int, default=64)
    parser.add_argument("-lr", type=float, default=0.01)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.RandomHorizontalFlip(p=0.5),
    #     transforms.RandomRotation(degrees=15),
    #     #transforms.ColorJitter(brightness=0.2, contrast=0.2),
    # ])
    # transform = Compose([RandomRotation(degrees=5)])
    
    train_data, train_label, test_data, test_label = dataloader.read_bci_data()
    train_dataset = BCIDataset(train_data, train_label)  #, transform=transform)
    test_dataset = BCIDataset(test_data, test_label)  #, transform=None)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = EEGNet()
    criterion = nn.CrossEntropyLoss()  #CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.01)  #, momentum=0.9)

    model.to(device)
    criterion.to(device)

    train_acc_list, train_loss_list, test_acc_list = train(model, train_loader, criterion, optimizer, args)

    plot_train_acc(train_acc_list, args.num_epochs)
    plot_train_loss(train_loss_list, args.num_epochs)
    plot_test_acc(test_acc_list, args.num_epochs)