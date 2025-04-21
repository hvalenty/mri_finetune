import os
import pandas as pd
import numpy as np

import torch
import torch.utils.data as data

from sampleaugment import RandomRotate, RandomTranslate, RandomFlip, ToTensor, Compose, RandomAffine
from torchvision import transforms

INPUT_DIM = 224
MAX_PIXEL_VAL = 255
MEAN = 58.09
STDDEV = 49.73

class MRData():
    """This class loads the MRnet dataset from the ./images directory."""

    def __init__(self, stage="train", transform=None, weights=None):
        self.planes = ['axial', 'coronal', 'sagittal']
        self.records = None
        self.image_path = {}

        if stage == 'train':
            self.records = pd.read_csv('./images/train.csv', header=None, names=['id', 'label'])
            for plane in self.planes:
                self.image_path[plane] = './images/train/{}/'.format(plane)
        elif stage == 'val':
            transform = None
            self.records = pd.read_csv('./images/val.csv', header=None, names=['id', 'label'])
            for plane in self.planes:
                self.image_path[plane] = './images/val/{}/'.format(plane)
        else:
            transform = None
            self.records = pd.read_csv('./images/test.csv', header=None, names=['id', 'label'])
            for plane in self.planes:
                self.image_path[plane] = './images/test/{}/'.format(plane)

        self.transform = transform
        
        self.records['id'] = self.records['id'].map(
            lambda i: '0' * (3 - len(str(i))) + str(i))

        # Convert labels to a dict: id → label
        self.labels = dict(zip(self.records['id'], self.records['label']))
        
        print(self.records['id'].tolist())

        # Image paths for each plane
        self.paths = {}
        for plane in self.planes:
            self.paths[plane] = [self.image_path[plane] + filename + '.npy' for filename in self.records['id'].tolist()]

        # Class statistics
        label_values = list(self.labels.values())
        print("Unique labels found in dataset:", sorted(set(label_values)))
        print("Number of classes:", len(set(label_values)))

        class_counts = pd.Series(label_values).value_counts().sort_index()
        num_classes = class_counts.shape[0]

        print('Class distribution:')
        for i, count in enumerate(class_counts):
            print(f"Class {i}: {count} samples")

        class_weights = 1.0 / class_counts
        class_weights = class_weights / class_weights.sum() * num_classes
        self.weights = torch.FloatTensor(class_weights.tolist())
        print('Class weights for loss are:', self.weights)

        print(f'Total samples: {len(self.records)} | Num classes: {num_classes}')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        img_raw = {}
        record_id = self.records.iloc[index]['id']

        for plane in self.planes:
            img_raw[plane] = np.load(self.paths[plane][index])
            img_raw[plane] = self._resize_image(img_raw[plane])

        label = torch.tensor(int(self.labels[record_id])).long()
        return [img_raw[plane] for plane in self.planes], label

    def _resize_image(self, image):
        pad = int((image.shape[2] - INPUT_DIM) / 2)
        image = image[:, pad:-pad, pad:-pad]
        image = (image - np.min(image)) / (np.max(image) - np.min(image)) * MAX_PIXEL_VAL
        image = (image - MEAN) / STDDEV

        if self.transform:
            image = self.transform(image)
        else:
            image = np.stack((image,) * 3, axis=1)

        return torch.FloatTensor(image)

def load_data():
    augments = Compose([
        transforms.Lambda(lambda x: torch.Tensor(x)),
        RandomRotate(25),
        RandomTranslate([0.11, 0.11]),
        RandomFlip(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1, 1).permute(1, 0, 2, 3)),
    ])

    print('Loading Train Dataset of ACL task...')
    train_data = MRData(stage='train', transform=augments)
    train_loader = data.DataLoader(train_data, batch_size=1, num_workers=2, shuffle=True)

    print('Loading Validation Dataset of ACL task...')
    val_data = MRData(stage='val')
    val_loader = data.DataLoader(val_data, batch_size=1, num_workers=2, shuffle=False)

    print('Loading Testing Dataset of ACL task...')
    test_data = MRData(stage='test')
    test_loader = data.DataLoader(test_data, batch_size=1, num_workers=2, shuffle=False)

    return train_loader, val_loader, test_loader, train_data.weights, val_data.weights, test_data.weights
