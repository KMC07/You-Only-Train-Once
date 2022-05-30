import torch
import torchvision.transforms as transforms
from torch.utils.data import random_split
import os
from torchvision.datasets import ImageFolder
import tarfile
from torchvision.datasets.utils import download_url

BATCHSIZE = 128
CIFAR_data_dir = './data/cifar10'
SHAPES_train_dir = './data/shapes3dtrain'
SHAPES_test_dir = './data/shapes3dtest'

def prepare_CIFAR_dataset(download=True):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize(32)])

    if download:
        dataset_url = "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz"
        download_url(dataset_url, '.')
        
        # Extract from archive
        with tarfile.open('./cifar10.tgz', 'r:gz') as tar:
            tar.extractall(path='./data')

    trainset = ImageFolder(CIFAR_data_dir + '/train', transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCHSIZE, shuffle=True, num_workers=2)

    testset = ImageFolder(CIFAR_data_dir + '/test', transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCHSIZE, shuffle=False, num_workers=2)

    return trainloader, testloader


def prepare_SHAPES_dataset(download=True):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize(32)])

    batch_size = 128
    
    if download:
        #the h5py file must be downloaded and put into the data directory
        import h5py
        dataset = h5py.File('./data/3dshapes.h5', 'r')
        images = dataset['images']  # array shape [480000,64,64,3], uint8 in range(256)

        from PIL import Image

        #create the train dataset
        i = 0
        for img in images:
            im1 = Image.fromarray(img)
            im1.save(f"data/shapes3dtrain/train/{i}.png")
            i += 1
            if i == 100000:
                break

        #create the test dataset
        i = 100000
        for img in images:
            img1 = Image.fromarray(img)
            img1.save(f"data/shapes3dtest/train/{i}.png")
            i += 1
            if i == 110000:
                break
    
    images_train = ImageFolder('data/shapes3dtrain', transform=transform)
    images_test = ImageFolder('data/shapes3dtest', transform=transform)

    trainloader = torch.utils.data.DataLoader(images_train, batch_size=batch_size, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(images_test, batch_size=batch_size, shuffle=True, num_workers=0)

    return trainloader, testloader