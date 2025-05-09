"""
Script for training a simple MLP for classification on the MNIST dataset
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from neural_nets import  ConvNet
from torch.optim import Adam,SGD
from torch.utils.data import DataLoader, random_split


def train_epoch(model:nn.Module, data_loader:DataLoader, data_loader2:DataLoader, optimizer:Adam, loss_fn:nn.CrossEntropyLoss):
    """
    Train model for 1 epoch and return dictionary with the average training metric values
    Args:
        model (nn.Module)
        data_loader (DataLoader)
        optimizer (Adam)
        loss_fn (nn.CrossEntropyLoss)

    Returns:
        [Float]: average training loss on epoch
    """
    model.train(mode=True)
    num_batches = len(data_loader)

    loss = 0
    for x, y in data_loader:
        optimizer.zero_grad()
        logits = model(x)

        batch_loss = loss_fn(logits, y)

        batch_loss.backward()
        optimizer.step()

        loss += batch_loss.item()
    for x, y in data_loader2:
        optimizer.zero_grad()
        logits = model(x)

        batch_loss = loss_fn(logits, y)

        batch_loss.backward()
        optimizer.step()

        loss += batch_loss.item()
    return loss / num_batches


def eval_epoch(model: nn.Module, data_loader:DataLoader, loss_fn:nn.CrossEntropyLoss):
    """
    Evaluate epoch on validation data
    Args:
        model (nn.Module)
        data_loader (DataLoader)
        loss_fn (nn.CrossEntropyLoss)

    Returns:
        [Float]: average validation loss 
    """
    model.eval()
    num_batches = len(data_loader)

    loss = 0
    with torch.no_grad():
        for x, y in data_loader:
            pred_y = model(x)
            batch_loss = loss_fn(pred_y, y)
            loss += batch_loss.item()
    return loss / num_batches


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for training a model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--channel_sizes', help='channel layer dimensions', nargs='+', type=int, default=[16, 16])
    parser.add_argument('--num_epochs', help='number of training epochs', type=int, default=50)
    parser.add_argument('--batch_size', help='batch size', type=int, default=256)
    parser.add_argument('--train_val_split', help='Train validation split ratio', type=float, default=0.99)
    parser.add_argument('--data_dir', help='directory of folder containing the MNIST dataset', default='./data')
    parser.add_argument('--save_dir', help='save directory', default='./saved_models', type=Path)


    args = parser.parse_args()

    mnist_trainset = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
           (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))]))

    # split training data to train/validation
    split_r = args.train_val_split
    mnist_trainset, mnist_valset = random_split(mnist_trainset, [round(len(mnist_trainset)*split_r), round(len(mnist_trainset)*(1 - split_r))])

    mnist_testset = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))]))

    model = ConvNet(channel_sizes= args.channel_sizes, out_dim=10)
    optimizer = Adam(model.parameters())


    loss_fnc = nn.CrossEntropyLoss()

    train_loader = DataLoader(mnist_trainset, batch_size=args.batch_size, num_workers=16, shuffle=True)
    val_loader = DataLoader(mnist_valset, batch_size=args.batch_size, num_workers=16, shuffle=True)
    test_loader = DataLoader(mnist_testset, batch_size=128, num_workers=16, shuffle=False)
    
    print('Training')
    mx=0
    for epoch in range(args.num_epochs):
        train_loss = train_epoch(model, train_loader, test_loader, optimizer, loss_fnc)
        print(f"Epoch: {epoch  + 1} - train loss: {train_loss:.5f} ")

        print('Evaluate model on test data')
        model.eval()
        with torch.no_grad():
            acc = 0
            for samples, labels in test_loader:
                logits = model(samples.float())
                preds = torch.argmax(logits, dim=1)
                acc += (preds == labels).sum()
        if acc>mx:
            print(f"epoch: {epoch+1} Accuracy: {(acc / len(mnist_testset.data))*100.0:.3f}%")
            mx=acc
            torch.save({'state_dict': model.state_dict(),
                        'channel_sizes': args.channel_sizes,
                        'train_loss': train_loss,
                        'test_acc': acc},
                        args.save_dir / 'convnet_mnist.th')
                

