import torch
from torchvision import datasets, transforms

def get_cifar10_loader(batch_size,num_worker):
    transform_train=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32,padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914,0.4822,0.4465)
                             ,std=(0.2470,0.2435,0.2616))
    ])

    transform_test=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914,0.4822,0.4465)
                             ,std=(0.2470,0.2435,0.2616))
    ])

    train_set=datasets.CIFAR10(root='./data',train=True,download=True,transform=transform_train)
    test_set=datasets.CIFAR10(root='./data',train=False,download=True,transform=transform_test)

    train_loader=torch.utils.data.DataLoader(train_set,batch_size=batch_size,shuffle=True,num_workers=num_worker,pin_memory=True)
    test_loader=torch.utils.data.DataLoader(test_set,batch_size=batch_size,shuffle=False,num_workers=num_worker,pin_memory=True)

    return train_loader,test_loader