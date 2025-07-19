import torchvision as tv
import torch.utils.data as data

def get_mnist_data(cfg):
    dataset = tv.datasets.MNIST("./datas", download=True, transform=tv.transforms.ToTensor())
    train, val = data.random_split(dataset, [55000, 5000])
    return train, val