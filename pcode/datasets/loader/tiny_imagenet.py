# Framework adapted from FedDF; see LICENSE and NOTICE.
# Modified for the FedSCAD release: retain the shared-anchor training path.

import os
import torch.utils.data as data
from PIL import Image
from torchvision.datasets.utils import download_and_extract_archive

class TinyImageNet(data.Dataset):

    def __init__(self, root, split='train', transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        if download:
            self._download()
        self.data = []
        self.targets = []
        wnids_path = os.path.join(self.root, 'wnids.txt')
        if not os.path.exists(wnids_path):
            train_dir = os.path.join(self.root, 'train')
            if not os.path.exists(train_dir):
                raise RuntimeError('Dataset not found. You can use download=True to download it')
            self.wnids = sorted(os.listdir(train_dir))
        else:
            with open(wnids_path, 'r') as f:
                self.wnids = sorted([x.strip() for x in f.readlines()])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.wnids)}
        if len(self.wnids) > 0:
            print(f'[TinyImageNet] Class mapping check: 0 -> {self.wnids[0]}, 1 -> {self.wnids[1]}, ... {len(self.wnids) - 1} -> {self.wnids[-1]}')
        if self.split == 'train':
            train_dir = os.path.join(self.root, 'train')
            for cls in self.wnids:
                cls_dir = os.path.join(train_dir, cls, 'images')
                if not os.path.isdir(cls_dir):
                    continue
                for img_name in os.listdir(cls_dir):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.data.append(os.path.join(cls_dir, img_name))
                        self.targets.append(self.class_to_idx[cls])
        elif self.split == 'val':
            val_dir = os.path.join(self.root, 'val')
            img_dir = os.path.join(val_dir, 'images')
            annot_file = os.path.join(val_dir, 'val_annotations.txt')
            with open(annot_file, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    img_name = parts[0]
                    cls = parts[1]
                    if cls in self.class_to_idx:
                        self.data.append(os.path.join(img_dir, img_name))
                        self.targets.append(self.class_to_idx[cls])
        elif self.split == 'test':
            pass

    def _download(self):
        if os.path.exists(os.path.join(self.root, 'wnids.txt')):
            print('Files already downloaded and verified')
            return
        lock_dir = os.path.abspath(os.path.dirname(self.root))
        os.makedirs(lock_dir, exist_ok=True)
        lock_file = os.path.join(lock_dir, 'tiny_imagenet.lock')

        def _try_acquire_lock():
            try:
                fd = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                return False
            try:
                os.write(fd, b'locking')
            finally:
                os.close(fd)
            return True
        import time
        while not _try_acquire_lock():
            if os.path.exists(os.path.join(self.root, 'wnids.txt')):
                print('Download completed by another process.')
                return
            print('Another process is downloading the dataset. Waiting...')
            time.sleep(1)
        try:
            url = 'https://cs231n.stanford.edu/tiny-imagenet-200.zip'
            filename = 'tiny-imagenet-200.zip'
            download_root = lock_dir
            print(f'Downloading {url} to {os.path.join(download_root, filename)}')
            download_and_extract_archive(url, download_root, filename=filename)
            extracted_dir = os.path.join(download_root, 'tiny-imagenet-200')
            if os.path.exists(self.root):
                import shutil
                shutil.rmtree(self.root)
            os.rename(extracted_dir, self.root)
            print('Done!')
        finally:
            if os.path.exists(lock_file):
                os.remove(lock_file)

    def __getitem__(self, index):
        img_path = self.data[index]
        target = self.targets[index]
        with open(img_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return (img, target)

    def __len__(self):
        return len(self.data)
