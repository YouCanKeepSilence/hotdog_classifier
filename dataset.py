import torch
from torch.utils import data
from torchvision import datasets, transforms

_transform = transforms.Compose([transforms.Resize(255),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor()])


def _get_loader(path, transform, batch_size=32, shuffle=True):
    dataset = datasets.ImageFolder(path, transform=transform)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def get_train_loader(path='./data/train'):
    return _get_loader(path, _transform)


def get_test_loader(path='./data/test'):
    return _get_loader(path, _transform)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    loader = get_train_loader()
    print(len(loader))
    images, labels = next(iter(loader))
    plt.imshow(images[1].permute((1, 2, 0)))
    print(f'Is hot-dog: {labels[1].item() == 0}')
    plt.show()