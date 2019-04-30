from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import numpy as np
from net import MemoryBuffer

#########################################################
# credit assignment: based on the PyTorch MNIST example #
# https://github.com/pytorch/examples/tree/master/mnist #
#########################################################


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

        self.memory_buffer = MemoryBuffer()
        self.use_memory_buffer = False

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def clone_parameters(self, other):
        self.conv1.weight.data = other.conv1.weight.data.clone()
        self.conv1.bias.data = other.conv1.bias.data.clone()
        self.conv2.weight.data = other.conv2.weight.data.clone()
        self.conv2.bias.data = other.conv2.bias.data.clone()
        self.fc1.weight.data = other.fc1.weight.data.clone()
        self.fc1.bias.data = other.fc1.bias.data.clone()
        self.fc2.weight.data = other.fc2.weight.data.clone()
        self.fc2.bias.data = other.fc2.bias.data.clone()

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        if not model.use_memory_buffer:
            loss.backward()
        else:
            model.memory_buffer.store_memory(data, target, loss.item())
            model.memory_buffer.compute_loss_and_backward_pass_for_random_memory(model, F.nll_loss)

        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct_full = pred.eq(target.view_as(pred))
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return correct_full.numpy()

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=10000, metavar='N',  # TODO changed from 1000
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',  # TODO changed from 10
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',  # TODO changed from 1
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N', # TODO changed from 10
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_dataset = datasets.MNIST('../data', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    train_dataset = sorted(train_dataset, key=lambda x: x[1])  # sort dataset by label
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=args.batch_size, shuffle=False, **kwargs)

    test_dataset = datasets.MNIST('../data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]))
    test_dataset = sorted(test_dataset, key=lambda x: x[1])  # sort dataset by label
    test_loader = torch.utils.data.DataLoader(test_dataset,
        batch_size=args.test_batch_size, shuffle=False, **kwargs)

    model = Net()
    model_mr = Net()
    model_mr.clone_parameters(model)

    history_correct_full = outer_train(args, model.to(device), device, train_loader, test_loader, "mnist_cnn.pt")

    model_mr.use_memory_buffer = True
    model_mr.memory_buffer.max_size = 120
    history_correct_full_mr = outer_train(args, model_mr.to(device), device, train_loader, test_loader, "mnist_cnn_mr.pt")

    def moving_average(a, window_size):
        new_a = np.empty_like(a, dtype=np.double)
        for i in range(len(a) - window_size):
            new_a[i] = np.mean(a[i:i + window_size])
        return new_a[:i]

    fig = plt.figure()
    ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
    ax.set_xlabel('Test samples (ordered by label)')
    ax.set_ylabel('Classification accuracy')

    ax.plot(moving_average(history_correct_full[0], 200), label='bp 0', color='C8')
    ax.plot(moving_average(history_correct_full[-1], 200), label=f'bp {args.epochs - 1}', color='C1')
    ax.plot(moving_average(history_correct_full_mr[0], 200), label='mr 0', color='C6')
    ax.plot(moving_average(history_correct_full_mr[-1], 200), label=f'mr {args.epochs - 1}', color='C3')
    for i in range(0, 11):
        ax.axvline(i * 1000, color='k', lw=0.5)
    ax.legend(fontsize=8)

    plt.savefig('../figures/mnist.png', dpi=300)

def outer_train(args, model, device, train_loader, test_loader, fn):

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    history_correct_full = []
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        correct_full = test(args, model, device, test_loader)
        history_correct_full.append(correct_full)

    if (args.save_model):
        torch.save(model.state_dict(), fn)

    return history_correct_full

if __name__ == '__main__':
    main()
