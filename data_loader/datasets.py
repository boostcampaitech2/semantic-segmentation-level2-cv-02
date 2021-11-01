from torchvision import transforms
from torch.utils.data import Dataset as Dset, DataLoader, ConcatDataset as ConcatDset
import cv2
import numpy as np
from pycocotools.coco import COCO
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import os
import json

category_names = [
    "Backgroud",
    "General trash",
    "Paper",
    "Paper pack",
    "Metal",
    "Glass",
    "Plastic",
    "Styrofoam",
    "Plastic bag",
    "Battery",
    "Clothing",
]


def get_classname(classID, cats):
    for i in range(len(cats)):
        if cats[i]["id"] == classID:
            return cats[i]["name"]
    return "None"


class BasicDataset(Dset):
    """
    Thrash Dset
    """

    def __init__(self, data_dir, ann_file, mode="train", transform=None):
        super().__init__()
        self.mode = mode
        self.transform = transform
        self.coco = COCO(ann_file)
        self.data_dir = data_dir

    def __getitem__(self, index: int):
        # dataset이 index되어 list처럼 동작
        image_id = self.coco.getImgIds(imgIds=index)
        image_infos = self.coco.loadImgs(image_id)[0]

        # cv2 를 활용하여 image 불러오기
        images = cv2.imread(os.path.join(self.data_dir, image_infos["file_name"]))
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32)
        images /= 255.0

        if self.mode in ("train", "val"):
            ann_ids = self.coco.getAnnIds(imgIds=image_infos["id"])
            anns = self.coco.loadAnns(ann_ids)

            # Load the categories in a variable
            cat_ids = self.coco.getCatIds()
            cats = self.coco.loadCats(cat_ids)

            # masks : size가 (height x width)인 2D
            # 각각의 pixel 값에는 "category id" 할당
            # Background = 0
            masks = np.zeros((image_infos["height"], image_infos["width"]))
            # General trash = 1, ... , Cigarette = 10
            anns = sorted(anns, key=lambda idx: len(idx["segmentation"][0]), reverse=False)
            for i in range(len(anns)):
                className = get_classname(anns[i]["category_id"], cats)
                pixel_value = category_names.index(className)
                masks[self.coco.annToMask(anns[i]) == 1] = pixel_value
            masks = masks.astype(np.int8)

            # transform -> albumentations 라이브러리 활용
            if self.transform is not None:
                transformed = self.transform(image=images, mask=masks)
                images = transformed["image"]
                masks = transformed["mask"]
            return images, masks, image_infos

        if self.mode == "test":
            # transform -> albumentations 라이브러리 활용
            if self.transform is not None:
                transformed = self.transform(image=images)
                images = transformed["image"]
            return images, image_infos

    def __len__(self) -> int:
        # 전체 dataset의 size를 return
        return len(self.coco.getImgIds())


class ResizedBasicDataset(Dset):
    """
    Thrash Dset
    """

    def __init__(self, data_dir="../input/resized_data_256", mode="train", transform=None):
        super().__init__()
        self.mode = mode
        self.transform = transform
        self.data_dir = data_dir
        origin_dataset_path = "../input/data"
        if mode == "train":
            image_config_path = origin_dataset_path + "/train.json"
        elif mode == "val":
            image_config_path = origin_dataset_path + "/val.json"
        with open(image_config_path) as json_file:
            self.data_json = json.load(json_file)
        self.file_names = list(
            map(lambda x: str(int(x["file_name"].split("/")[1].split(".")[0])), self.data_json["images"])
        )

    def __getitem__(self, index: int):
        # dataset이 index되어 list처럼 동작

        images = cv2.imread(os.path.join(self.data_dir, "image", str(self.file_names[index]) + ".jpg"))
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32)
        images /= 255.0

        masks = np.load(os.path.join(self.data_dir, "mask", str(index) + ".npy"))
        if self.transform is not None:
            transformed = self.transform(image=images, mask=masks)
            images = transformed["image"]
            masks = transformed["mask"]
        return images, masks, None

    def __len__(self) -> int:
        # 전체 dataset의 size를 return
        # return 3272
        return len(self.file_names)
        # return len(self.coco.getImgIds())


def ConcatDataset(datasets):
    return ConcatDset(datasets=datasets)
