import os
import copy
import librosa as li
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from torchvision import datasets, transforms
import IPython.display as ipd


def play_audiomnist(x):
    ipd.Audio(x.detach().cpu().numpy().flatten(), rate=16000)  # load a NumPy array


def plot_audiomnist(x, y, model):
    """
    Plot waveform and class scores side-by-side

    :param x:
    :param y:
    :param model:
    :return:
    """

    x = x.clone().detach().cpu()
    if isinstance(y, torch.Tensor):
        y = y.clone().detach().cpu()

    # use model to compute class scores and predicted label
    y_scores = torch.nn.functional.softmax(
        model(x.reshape(1, 1, 16000)), dim=-1
    )
    y_pred = y_scores.argmax()

    # initialize plot
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(6, 2.5))
    width = 0.5
    linewidth = 2.0

    # image plot
    axs[0].plot(x.squeeze().numpy(), 'k-')
    axs[0].set_xlabel('Sample idx')
    axs[0].set_ylabel('Amplitude')

    # class scores plot
    axs[1].bar(
        list(range(0, 10)),
        y_scores.flatten().detach().cpu().numpy(),
        width,
        color='black',
        label='class scores',
        edgecolor='black',
        linewidth=linewidth
    )

    # formatting
    fig.suptitle(f"True Label: {y}, Predicted Label: {y_pred}", y=1.1)
    axs[1].grid(False)
    axs[1].spines['left'].set_linewidth(linewidth)
    axs[1].set_xlim(-1, 10)
    axs[1].tick_params(bottom=True, left=True)
    axs[1].set_yscale('log')
    axs[1].set_xticks(list(range(0, 10)))
    sns.despine(bottom=True)
    plt.tight_layout()
    plt.show()


def train_audiomnist(model,
                     device,
                     train_loader,
                     test_loader,
                     epochs: int = 14,
                     save_model: bool = True
                     ):
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.001, momentum=0.9
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=1,
        gamma=1.0
    )

    criterion = torch.nn.CrossEntropyLoss()

    best_acc = 0.0

    for epoch in range(epochs):

        # track loss
        training_loss = 0.0
        validation_loss = 0

        # track accuracy
        correct = 0
        total = 0

        pbar = tqdm(train_loader, total=len(train_loader))

        model.train()
        for batch_idx, batch_data in enumerate(pbar):

            pbar.set_description(
                f'Epoch {epoch + 1}, batch {batch_idx + 1}/{len(train_loader)}')

            inputs, labels = batch_data

            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward + backward + optimize
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # sum training loss
            training_loss += loss.item()

        model.eval()
        with torch.no_grad():

            pbar = tqdm(test_loader, total=len(test_loader))
            for batch_idx, batch_data in enumerate(pbar):

                pbar.set_description(
                    f'Validation, batch {batch_idx + 1}/{len(test_loader)}')

                inputs, labels = batch_data

                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)

                loss = criterion(outputs, labels)

                # sum validation loss
                validation_loss += loss.item()

                # calculate validation accuracy
                preds = torch.max(outputs.data, 1)[1]

                total += labels.size(0)
                correct += (preds == labels).sum().item()

        # calculate final metrics
        validation_loss /= len(test_loader)
        training_loss /= len(train_loader)
        accuracy = 100 * correct / total

        # if best model thus far, save
        if accuracy > best_acc and save_model:
            print(f"New best accuracy: {accuracy}; saving model")
            best_model = copy.deepcopy(model.state_dict())
            best_acc = accuracy
            torch.save(
                best_model,
                "audionet.pt"
            )

        # update step-size scheduler
        scheduler.step()


def test_audiomnist(model, device, test_loader):

    model.to(device)
    model.eval()

    # track accuracy
    correct = 0
    total = 0

    with torch.no_grad():

        pbar = tqdm(test_loader, total=len(test_loader))
        for batch_idx, batch_data in enumerate(pbar):

            pbar.set_description(
                f'Validation, batch {batch_idx + 1}/{len(test_loader)}')

            inputs, labels = batch_data

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            # calculate validation accuracy
            preds = torch.max(outputs.data, 1)[1]

            total += labels.size(0)
            correct += (preds == labels).sum().item()

    accuracy = 100 * correct / total

    print(f"\nModel accuracy: {accuracy}")


def load_audiomnist(data_dir, train_batch_size: int = 64, test_batch_size: int = 128):

    audio_list = sorted(list(Path(data_dir).rglob(f'*.wav')))
    cache_list = sorted(list(Path(data_dir).rglob('*.pt')))  # check for cached dataset

    if len(cache_list) > 0:
        tx = torch.load(os.path.join(data_dir, 'audiomnist_tx.pt'))
        ty = torch.load(os.path.join(data_dir, 'audiomnist_ty.pt'))

    else:
        tx = torch.zeros((len(audio_list), 1, 16000))
        ty = torch.zeros(len(audio_list), dtype=torch.long)

        pbar = tqdm(audio_list, total=len(audio_list))

        for i, audio_fn in enumerate(pbar):
            pbar.set_description(
                f'Loading AudioMNIST ({os.path.basename(audio_fn)})')
            waveform, _ = li.load(audio_fn,
                                  mono=True,
                                  sr=16000,
                                  duration=1.0)
            waveform = torch.from_numpy(waveform)

            tx[i, :, :waveform.shape[-1]] = waveform
            ty[i] = int(os.path.basename(audio_fn).split("_")[0])

        torch.save(tx, os.path.join(data_dir, 'audiomnist_tx.pt'))
        torch.save(ty, os.path.join(data_dir, 'audiomnist_ty.pt'))

    # partition data
    tx_train, ty_train, tx_test, ty_test = [], [], [], []
    for i in range(10):

        idx = ty == i
        tx_i = tx[idx]
        ty_i = ty[idx]

        split = int(0.8 * len(tx_i))

        tx_train.append(tx_i[:split]), ty_train.append(ty_i[:split])
        tx_test.append(tx_i[split:]), ty_test.append(ty_i[split:])

    tx_train = torch.cat(tx_train, dim=0)
    ty_train = torch.cat(ty_train, dim=0)
    tx_test = torch.cat(tx_test, dim=0)
    ty_test = torch.cat(ty_test, dim=0)

    # create datasets
    train_data = torch.utils.data.TensorDataset(tx_train, ty_train)

    test_data = torch.utils.data.TensorDataset(tx_test, ty_test)

    # load data
    cuda_kwargs = {
        'num_workers': 1,
        'pin_memory': True,
        'shuffle': True
    } if torch.cuda.is_available() else {}

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=train_batch_size,
        **cuda_kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=test_batch_size,
        **cuda_kwargs
    )

    return train_loader, test_loader


def plot_mnist(x, y, model):
    """
    Plot grayscale image and class scores side-by-side

    :param x:
    :param y:
    :param model:
    :return:
    """

    x = x.clone().detach().cpu()
    if isinstance(y, torch.Tensor):
        y = y.clone().detach().cpu()

    # use model to compute class scores and predicted label
    y_scores = torch.nn.functional.softmax(
        model(x.reshape(1, 1, 28, 28)), dim=-1
    )
    y_pred = y_scores.argmax()

    # initialize plot
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(6, 2.5))
    width = 0.5
    margin = 0.0025
    linewidth = 2.0

    # image plot
    axs[0].imshow(x.squeeze().numpy(), cmap='gray')

    # class scores plot
    axs[1].bar(
        list(range(0, 10)),
        y_scores.flatten().detach().cpu().numpy(),
        width,
        color='black',
        label='class scores',
        edgecolor='black',
        linewidth=linewidth
    )

    # formatting
    fig.suptitle(f"True Label: {y}, Predicted Label: {y_pred}", y=1.1)
    axs[1].grid(False)
    axs[1].spines['left'].set_linewidth(linewidth)
    axs[1].set_xlim(-1, 10)
    axs[1].tick_params(bottom=True, left=True)
    axs[1].set_yscale('log')
    axs[1].set_xticks(list(range(0, 10)))
    sns.despine(bottom=True)
    plt.tight_layout()
    plt.show()


def train_mnist(
        model,
        device,
        train_loader,
        test_loader,
        epochs: int = 14,
        log_interval: int = 50,
        save_model: bool = True
):
    """
    Train a simple MNIST classifier. MNIST classification code adapted from
    https://github.com/pytorch/examples/blob/master/mnist/main.py

    :param model:
    :param device:
    :param train_loader:
    :param test_loader:
    :param epochs:
    :param log_interval:
    :param save_model:

    :return:
    """

    # configure optimization
    optimizer = optim.Adadelta(model.parameters(), lr=1.0)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

    for epoch in range(1, epochs + 1):

        # training step
        model.train()  # training mode
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))

        # validation step
        model.eval()  # evaluation mode
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

        scheduler.step()

    if save_model:
        torch.save(model.state_dict(), "../models/mnist_cnn.pt")


def test_mnist(model, device, test_loader):
    """
    Evaluate a simple MNIST classifier. MNIST classification code adapted from
    https://github.com/pytorch/examples/blob/master/mnist/main.py
    """

    model.eval()  # evaluation mode
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def load_mnist(train_batch_size: int = 64, test_batch_size: int = 1000):
    """
    Load MNIST dataset. MNIST classification code adapted from
    https://github.com/pytorch/examples/blob/master/mnist/main.py
    :return: train and test DataLoader objects
    """

    cuda_kwargs = {
        'num_workers': 1,
        'pin_memory': True,
        'shuffle': True
    } if torch.cuda.is_available() else {}

    # format image data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # download MNIST data
    train_data = datasets.MNIST(
        '../data',
        train=True,
        download=True,
        transform=transform
    )
    test_data = datasets.MNIST(
        '../data',
        train=False,
        transform=transform,
    )

    # load MNIST data
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=train_batch_size,
        **cuda_kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=test_batch_size,
        **cuda_kwargs
    )

    return train_loader, test_loader




