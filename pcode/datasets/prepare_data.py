# Framework adapted from FedDF; see LICENSE and NOTICE.
# Modified for the FedSCAD release: retain the shared-anchor training path.

import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import pcode.datasets.loader.tiny_imagenet as tiny_imagenet
'the entry for classification tasks.'

def _get_cifar(conf, name, root, split, transform, target_transform, download):
    is_train = split == 'train'
    if name == 'cifar10':
        dataset_loader = datasets.CIFAR10
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.201)) if not conf.use_fake_centering else transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    elif name == 'cifar100':
        dataset_loader = datasets.CIFAR100
        normalize = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) if not conf.use_fake_centering else transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    normalize = normalize if conf.pn_normalize else None
    if is_train:
        transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop((32, 32), 4), transforms.ToTensor()] + ([normalize] if normalize is not None else []))
    else:
        transform = transforms.Compose([transforms.ToTensor()] + ([normalize] if normalize is not None else []))
    return dataset_loader(root=root, train=is_train, transform=transform, target_transform=target_transform, download=download)

def _get_tiny_imagenet(conf, root, split, transform, target_transform, download=True):
    is_train = split == 'train'
    normalize = transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)) if conf.pn_normalize else None
    if is_train:
        transform = transforms.Compose([transforms.RandomCrop(64, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor()] + ([normalize] if normalize is not None else []))
    else:
        transform = transforms.Compose([transforms.ToTensor()] + ([normalize] if normalize is not None else []))
    return tiny_imagenet.TinyImageNet(root=root, split=split if split != 'test' else 'val', transform=transform, target_transform=target_transform, download=download)
'the entry for different supported dataset.'

def get_dataset(conf, name, datasets_path, split='train', transform=None, target_transform=None, download=True):
    root = os.path.join(datasets_path, name)
    if name in ('cifar10', 'cifar100'):
        return _get_cifar(conf, name, root, split, transform, target_transform, download)
    if name == 'tiny_imagenet':
        return _get_tiny_imagenet(conf, root, split, transform, target_transform, download)
    raise ValueError('This release supports CIFAR-10, CIFAR-100 and Tiny-ImageNet.')

def get_combine_dataset(*args, **kwargs):
    raise ValueError('Use --prepare_data combine; this entry is a legacy FedDF path.')
