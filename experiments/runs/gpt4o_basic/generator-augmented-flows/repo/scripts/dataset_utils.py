import torch
from torchvision import datasets, transforms

def get_cifar10(data_paths, batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = datasets.CIFAR10(root=data_paths['train'], train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=data_paths['test'], train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# Similar functions can be written for other datasets like ImageNet and CelebA.

