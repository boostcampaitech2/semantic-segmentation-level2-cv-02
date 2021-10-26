from pycocotools.coco import COCO
import albumentations as A
from albumentations.pytorch import ToTensorV2
from data_loader.custom_transform import Elastic_Transform
from albumentations import ElasticTransform
from albumentations import (
    PadIfNeeded,
    HorizontalFlip,
    VerticalFlip,
    CenterCrop,
    Crop,
    Compose,
    Transpose,
    RandomRotate90,
    ElasticTransform,
    GridDistortion,
    OpticalDistortion,
    RandomSizedCrop,
    OneOf,
    CLAHE,
    RandomBrightnessContrast,
    RandomGamma,
    HueSaturationValue,
    RGBShift,
    RandomBrightness,
    RandomContrast,
    Resize,
    MotionBlur,
    MedianBlur,
    GaussianBlur,
    GaussNoise,
    GridDropout,  # GridMask
    ChannelShuffle,
    CoarseDropout,  # Cutout
)
from albumentations.augmentations.crops.transforms import CropNonEmptyMaskIfExists


def Elastic_Transform():
    return A.Compose(
        [
            ElasticTransform(),
            ToTensorV2(),
        ]
    )



def BasicTransform():
    return A.Compose(
        [
            ToTensorV2(),
        ]
    )


def CustomTransform():
    return A.Compose(
        [
            CropNonEmptyMaskIfExists(height=256, width=256, p=0.5),
            Resize(512, 512),
            GridDropout(ratio=0.2, holes_number_x=5, holes_number_y=5, random_offset=True, p=0.5),
            ToTensorV2(),
        ]
    )
