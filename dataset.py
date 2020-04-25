import numpy as np
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


def get_train_test_data_for_svm(path='./data/train', test_multiplier=0.2):
    def _get_numpy_arrays_from_tensor_dataset(_dataset):
        x = np.zeros((len(_dataset), 224 * 224 * 3), dtype=np.float32)
        y = np.zeros(len(_dataset))
        for idx, (image, label) in enumerate(_dataset):
            x[idx][:] = image.flatten()
            y[idx] = label
        return x, y

    svm_transform = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        ])
    full_dataset = datasets.ImageFolder(path, transform=svm_transform)
    test_size = int(len(full_dataset) * test_multiplier)
    train_size = len(full_dataset) - test_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    x_train, y_train = _get_numpy_arrays_from_tensor_dataset(train_dataset)
    x_test, y_test = _get_numpy_arrays_from_tensor_dataset(test_dataset)
    return x_train, x_test, y_train, y_test


def get_score_data_for_svm(path='./data/test'):
    svm_transform = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        ])
    train_data = ImageFolderWithPaths(path, transform=svm_transform)
    x_score = np.zeros((len(train_data.imgs), 224 * 224 * 3), dtype=np.float32)
    labels = []
    for idx, (image, label, image_path) in enumerate(train_data):
        x_score[idx][:] = image.flatten()

    return x_score, labels


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # loader = get_train_loader()
    # images, labels = next(iter(loader))
    # plt.imshow(images[1].permute((1, 2, 0)))
    # print(f'Is hot-dog: {labels[1].item() == 1}')
    data = get_train_data_for_svm()
    # plt.imshow(data[0][0][0], cmap='hot')
    # plt.imshow(data[0][0][1], cmap='summer')
    # plt.imshow(data[0][0][2], cmap='winter')
    # print(data[0][0].shape)
    print('get')
    # plt.show()