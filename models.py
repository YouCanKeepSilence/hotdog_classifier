from torch import nn
from torch.nn import functional as F


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

    def forward(self, x):
        x = self.maxPooling(F.relu(self.conv1(x)))
        x = self.maxPooling(F.relu(self.conv2(x)))
        x = self.maxPooling(F.relu(self.conv3(x)))
        x = x.view(-1, self.pre_linear_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x


