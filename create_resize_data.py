import os
import json
import warnings

warnings.filterwarnings("ignore")

import torch
import cv2
from tqdm import tqdm

import numpy as np
from tqdm import tqdm

# 전처리를 위한 라이브러리
from pycocotools.coco import COCO
import torchvision
import torchvision.transforms as transforms

import albumentations as A

# 시각화를 위한 라이브러리
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

plt.rcParams["axes.grid"] = False

# GPU 사용 가능 여부에 따라 device 정보 저장
device = "cuda" if torch.cuda.is_available() else "cpu"


batch_size = 1
dataset_path = "../input/data"
anns_file_path = dataset_path + "/" + "train_all.json"
train_path = dataset_path + "/train.json"
val_path = dataset_path + "/val.json"

transform1 = A.Compose(
    [
        A.Resize(width=256, height=256),
    ]
)
transform2 = A.Compose(
    [
        A.Resize(width=512, height=512),
    ]
)
transform3 = A.Compose(
    [
        A.Resize(width=1024, height=1024),
    ]
)

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


# folder create
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Creating directory. " + directory)


createFolder("/opt/ml/segmentation/input/resized_data_256")
createFolder("/opt/ml/segmentation/input/resized_data_256/image")
createFolder("/opt/ml/segmentation/input/resized_data_256/mask")


createFolder("/opt/ml/segmentation/input/resized_data_512")
createFolder("/opt/ml/segmentation/input/resized_data_512/image")
createFolder("/opt/ml/segmentation/input/resized_data_512/mask")

createFolder("/opt/ml/segmentation/input/resized_data_1024")
createFolder("/opt/ml/segmentation/input/resized_data_1024/image")
createFolder("/opt/ml/segmentation/input/resized_data_1024/mask")


coco = COCO(anns_file_path)

with open(anns_file_path) as json_file:
    origin_data_json = json.load(json_file)
indexes = []
for i in origin_data_json["images"]:
    indexes.append(i["id"])

for index in tqdm(indexes):
    image_id = coco.getImgIds(imgIds=index)
    image_infos = coco.loadImgs(image_id)[0]

    images = cv2.imread(os.path.join(dataset_path, image_infos["file_name"]))
    images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32)
    images /= 255.0

    ann_ids = coco.getAnnIds(imgIds=image_infos["id"])
    anns = coco.loadAnns(ann_ids)

    cat_ids = coco.getCatIds()
    cats = coco.loadCats(cat_ids)

    masks = np.zeros((image_infos["height"], image_infos["width"]))

    anns = sorted(anns, key=lambda idx: len(idx["segmentation"][0]), reverse=False)
    for i in range(len(anns)):
        className = get_classname(anns[i]["category_id"], cats)
        pixel_value = category_names.index(className)
        masks[coco.annToMask(anns[i]) == 1] = pixel_value
    masks = masks.astype(np.int8)

    # transform -> albumentations 라이브러리 활용
    transformed_256 = transform1(image=images, mask=masks)
    images_256 = transformed_256["image"]
    masks_256 = transformed_256["mask"]

    plt.imsave(f"../input/resized_data_256/image/{index}.jpg", images_256)
    np.save(f"../input/resized_data_256/mask/{index}", masks_256)

    transformed_512 = transform2(image=images, mask=masks)
    images_512 = transformed_512["image"]
    masks_512 = transformed_512["mask"]

    plt.imsave(f"../input/resized_data_512/image/{index}.jpg", images_512)
    np.save(f"../input/resized_data_512/mask/{index}", masks_512)

    transformed_1024 = transform3(image=images, mask=masks)
    images_1024 = transformed_1024["image"]
    masks_1024 = transformed_1024["mask"]

    plt.imsave(f"../input/resized_data_1024/image/{index}.jpg", images_1024)
    np.save(f"../input/resized_data_1024/mask/{index}", masks_1024)
