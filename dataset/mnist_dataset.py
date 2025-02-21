import torch.utils.data as data
import torchvision as tv

def get_mnist_data(args):
    dataset = tv.datasets.MNIST("./datas", download=True, transform=tv.transforms.ToTensor())
    train, val = data.random_split(dataset, [55000, 5000])
    train_loader = data.DataLoader(train, batch_size=args.batch_size, num_workers=args.num_workers)
    val_loader = data.DataLoader(val, batch_size=args.batch_size, num_workers=args.num_workers)
    return train_loader, val_loader