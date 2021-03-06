import datetime

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import dump, load

import models
import dataset


def write_net_with_ml_score(net, classifier):
    with open('test_labels.txt', 'w') as file:
        net.eval()
        if torch.cuda.is_available():
            net.cuda()
        loader = dataset.get_score_loader_for_net()
        with torch.no_grad():
            # because we have fake label here
            for x, _, file_name in loader:
                # forward
                if torch.cuda.is_available():
                    x = x.cuda()
                out = net(x)
                pred = classifier.predict(out.cpu())
                file.writelines([f'{file_name[i].split("/")[-1]} {pred[i].astype(int)}\n' for i in range(x.size(0))])


def write_net_result(model):
    with open('net_vgg_res.txt', 'w') as file:
        model.eval()
        test_loader = dataset.get_score_loader_for_net()
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


def test_net(model, criterion, test_loader):
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        val_acc = 0
        for x, y in test_loader:
            # forward
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
            out = model(x)
            loss = criterion(out, y)
            # 1 means axis 1
            _, pred = torch.max(out.data, 1)
            val_acc += pred.eq(y).sum().item() / y.size(0)  # (pred == y).numpy().mean()
            val_loss += loss.item()
        val_loss /= len(test_loader)
        val_acc /= len(test_loader)
    model.train()
    return val_loss, val_acc


def train_net(model_class, batch_size=128):
    net = model_class
    writer = SummaryWriter(f'./logs/{type(net).__name__}-{datetime.datetime.now()}')
    if torch.cuda.is_available():
        net.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())
    train_loader, test_loader = dataset.get_train_test_loaders_for_net(batch_size=batch_size)
    epochs = 20
    for e in range(epochs):
        net.train()
        for i, data in enumerate(train_loader):
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
                val_loss, val_acc = test_net(net, criterion, test_loader)
                print(f'Epoch: {e}, Iteration: {i} \n'
                      f'Loss: {loss.item()} , Acc: {acc} \n'
                      f'Val loss: {val_loss} , Val Acc: {val_acc} \n'
                      f'-----------------------------------------\n'
                      )
                writer.add_scalar('Train/Acc', acc, e * len(train_loader) + i)
                writer.add_scalar('Train/Loss', loss.item(), e * len(train_loader) + i)
                writer.add_scalar('Val/Loss', val_loss, e * len(train_loader) + i)
                writer.add_scalar('Val/Acc', val_acc, e * len(train_loader) + i)

    torch.save(net.state_dict(), f'{type(net).__name__}.model')
    print(f'{type(net).__name__} saved')


def train_ml_classifier(classifier):
    x_train, x_test, y_train, y_test = dataset.get_train_test_data_for_svm()
    # fit model
    print(f'Start learn {type(classifier).__name__} {datetime.datetime.now()}')
    classifier.fit(x_train, y_train)
    print(f'Finished learn {type(classifier).__name__} {datetime.datetime.now()}')
    y_pred = classifier.predict(x_test)
    # calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print('Model accuracy is: ', accuracy)
    dump(classifier, f'simple_{type(classifier).__name__}.joblib')
    print(f'{type(classifier).__name__} saved')


def pre_process_data_through_net(data_loader, net, feature_count):
    if torch.cuda.is_available():
        net.cuda()
    net.eval()
    with torch.no_grad():
        X, Y = np.zeros((len(data_loader.dataset), feature_count)), np.zeros(len(data_loader.dataset))
        for idx, (x, y) in enumerate(data_loader):
            if torch.cuda.is_available():
                x = x.cuda()
            out = net(x)
            X[idx * out.size(0): (idx + 1) * out.size(0)][:] = out.cpu()
            Y[idx * y.size(0): (idx + 1) * y.size(0)][:] = y
        return X, Y


def train_vgg_featured_ml_classifier(classifier, batch_size=64):
    vgg = models.TailedVGG16Features()

    train_loader, test_loader = dataset.get_train_test_loaders_for_net(batch_size=batch_size)

    # pre-processing for classifier
    x_test, y_test = pre_process_data_through_net(test_loader, vgg, 4096)
    x_train, y_train = pre_process_data_through_net(train_loader, vgg, 4096)
    # fit model
    print(f'Start learn {type(classifier).__name__} {datetime.datetime.now()}')
    classifier.fit(x_train, y_train)
    print(f'Finished learn {type(classifier).__name__} {datetime.datetime.now()}')
    y_pred = classifier.predict(x_test)
    # calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print('Model accuracy is: ', accuracy)
    dump(classifier, f'featured_vgg_{type(classifier).__name__}.joblib')
    print(f'Featured VGG {type(classifier).__name__} saved')
    print('Writing results...')
    write_net_with_ml_score(vgg, classifier)


if __name__ == '__main__':
    print(f'Start learn net {datetime.datetime.now()}')
    c_net = models.TailedVGG16()
    train_net(c_net, batch_size=32)
    print(f'Finished learn net {datetime.datetime.now()}')
    # ml_class = SVC(kernel='linear', probability=True, random_state=42)
    # ml_class = RandomForestClassifier(10000, random_state=42)
    # train_ml_classifier(ml_class)
    # train_vgg_featured_ml_classifier(ml_class)
