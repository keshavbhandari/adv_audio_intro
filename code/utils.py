import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import seaborn as sns
import matplotlib.pyplot as plt


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




