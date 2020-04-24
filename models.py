from torch import nn
from torch.nn import functional as F
from torchvision import models


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 9, kernel_size=3)
        self.conv2 = nn.Conv2d(9, 27, kernel_size=3)
        self.conv3 = nn.Conv2d(27, 81, kernel_size=3)
        self.maxPooling = nn.MaxPool2d(kernel_size=3)
        self.pre_linear_size = 81 * 7 * 7
        self.fc1 = nn.Linear(self.pre_linear_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 2)
        self.dropout05 = nn.Dropout(0.5)
        self.dropout025 = nn.Dropout(0.25)

    def forward(self, x):
        x = self.maxPooling(F.relu(self.conv1(x)))
        x = self.maxPooling(F.relu(self.conv2(x)))
        x = self.maxPooling(F.relu(self.conv3(x)))
        x = x.view(-1, self.pre_linear_size)
        x = F.relu(self.fc1(x))
        x = self.dropout05(x)
        x = F.relu(self.fc2(x))
        x = self.dropout025(x)
        x = F.log_softmax(self.fc3(x), dim=1)
        return x


class TailedVGG16(nn.Module):
    def __init__(self):
        super(TailedVGG16, self).__init__()
        self.vgg16 = models.vgg16(pretrained=True)
        self.features = nn.Sequential(*list(self.vgg16.features)[:-1])
        self.classifier = nn.Linear(100352, 2)

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 100352)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

