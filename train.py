import torch

from models import CNN
from dataset import get_train_loader
import torch.nn as nn
import torch.optim as optim
import tqdm

if __name__ == '__main__':
    net = CNN()
    if torch.cuda.is_available():
        net.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())
    train_loader = get_train_loader()
    epochs = 20
    for epoch in tqdm.tqdm(range(epochs)):
        running_loss = 0.0
        acc = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.long().cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            _, pred = torch.max(outputs.data, 1)
            acc = pred.eq(labels).sum().item() / labels.size(0)  # (pred == y).numpy().mean()
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

        print(f' Acc: {acc * 100} %')

    print('Finished Training')