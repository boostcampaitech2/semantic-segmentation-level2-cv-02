from pycocotools.coco import COCO
import albumentations as A
from albumentations.pytorch import ToTensorV2

def BasicTransform():
    return A.Compose([ToTensorV2()])