from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import cv2
import numpy as np
from pycocotools.coco import COCO
import albumentations as A
from albumentations.pytorch import ToTensorV2

# from datasets import BasicDataset
# import os
# import matplotlib.pyplot as plt

# collate_fn needs for batch
def collate_fn(batch):
    return tuple(zip(*batch))

class BasicDataLoader(DataLoader):
    """
    Thrash DataLoader 
    """
    def __init__(self, dataset, batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        super().__init__(
            dataset=self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )


if __name__ == "__main__":
    category_names = ['Backgroud', 'General trash', 'Paper', 'Paper pack', 'Metal',
    'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing']
    train_transform = A.Compose([
                            ToTensorV2()
                            ])
    dataset_path  = '../input/data'
    train_path = dataset_path + '/train.json'
    train_dataset = BasicDataset(data_dir=dataset_path, ann_file=train_path, mode='train', transform=train_transform)
    train_loader = BasicDataLoader(dataset=train_dataset, 
                                           batch_size=4,
                                           shuffle=True,
                                           num_workers=4)
    for imgs, masks, image_infos in train_loader:
        image_infos = image_infos[0]
        temp_images = imgs
        temp_masks = masks
        break

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 12))

    print('image shape:', list(temp_images[0].shape))
    print('mask shape: ', list(temp_masks[0].shape))
    print('Unique values, category of transformed mask : \n', [{int(i),category_names[int(i)]} for i in list(np.unique(temp_masks[0]))])

    ax1.imshow(temp_images[0].permute([1,2,0]))
    ax1.grid(False)
    ax1.set_title("input image : {}".format(image_infos['file_name']), fontsize = 15)

    ax2.imshow(temp_masks[0])
    ax2.grid(False)
    ax2.set_title("masks : {}".format(image_infos['file_name']), fontsize = 15)

    plt.show()
    plt.savefig("sample.png")