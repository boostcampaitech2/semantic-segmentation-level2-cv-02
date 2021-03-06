from .basic_model import *
from .fcn import *
from .unet import *
from .hrnet import *
from .deeplab import *
from .hrnet_ocr import *

__all__ = [
    "BasicModel",
    "BasicModel2",
    "FCN8s",
    "FCN16s",
    "FCN32s",
    "UNet",
    "UnetPlusPlus",
    "HRNet",
    "HRNet_OCR",
    "UNet_2Plus",
    "UNet_3Plus",
    "UNet_3Plus_DeepSup",
    "UNet_3Plus_DeepSup_CGM",
    "DeepLabV3",
    "DeepLabV3Plus",
]
