from pycocotools.coco import COCO
import albumentations as A
from albumentations.pytorch import ToTensorV2
from data_loader.custom_transform import Elastic_Transform


def ElasticTransform():
    return A.Compose(
        [
            Elastic_Transform(p=0.5),
            ToTensorV2(),
        ]
    )


def BasicTransform():
    return A.Compose(
        [
            ToTensorV2(),
        ]
    )
