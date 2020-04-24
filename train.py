import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

from models import CNN, TailedVGG16
from dataset import get_train_loader, get_test_loader
import torch.nn as nn
import torch.optim as optim


def write_result(model):
    with open('net_vgg_res.txt', 'w') as file:
        model.eval()
        test_loader = get_test_loader()
        with torch.no_grad():
            # because we have fake label here
            for x, _, file_name in test_loader:
                # forward
                x = x.cuda()
                out = model(x)
                # 1 means axis 1
                _, pred = torch.max(out.data, 1)
                file.writelines([f'{file_name[i]} {pred[i].item()}\n' for i in range(x.size(0))])

        model.train()


def train():
    net = TailedVGG16()
    print(type(net).__name__)
    writer = SummaryWriter(f'./logs/{type(net).__name__}-{datetime.datetime.now()}')
    if torch.cuda.is_available():
        net.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())
    train_loader = get_train_loader()
    epochs = 20
    for e in range(epochs):
        net.train()
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [x, y]
            x, y = data
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(x)
            loss = criterion(outputs, y)
            _, pred = torch.max(outputs.data, 1)
            acc = pred.eq(y).sum().item() / y.size(0)  # (pred == y).numpy().mean()
            loss.backward()
            optimizer.step()

            # print statistics
            if i % 1 == 0:
                print(f'Epoch: {e}, Iteration: {i} \n'
                      f'X: {x.size()}. Y: {y.size()} \n'
                      f'Loss: {loss.item()} , Acc: {acc} \n'
                      f'-----------------------------------------\n'
                      )
                writer.add_scalar('Train/Acc', acc, e * len(train_loader))
                writer.add_scalar('Train/Loss', loss.item(), e * len(train_loader) + i)

    write_result(net)


if __name__ == '__main__':
    train()