import torch
from torch.utils import data
from torchvision import datasets, transforms

_transform = transforms.Compose([transforms.Resize(255),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor()])

_default_batch = 56


class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


def _get_loader(path, transform, batch_size=_default_batch, shuffle=True):
    dataset = datasets.ImageFolder(path, transform=transform)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def get_train_loader(path='./data/train'):
    return _get_loader(path, _transform, batch_size=_default_batch)


def get_test_loader(path='./data/test'):
    dataset = ImageFolderWithPaths(path, transform=_transform)
    return torch.utils.data.DataLoader(dataset, batch_size=_default_batch, shuffle=False)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    loader = get_train_loader()
    images, labels = next(iter(loader))
    plt.imshow(images[1].permute((1, 2, 0)))
    print(f'Is hot-dog: {labels[1].item() == 1}')
    plt.show()