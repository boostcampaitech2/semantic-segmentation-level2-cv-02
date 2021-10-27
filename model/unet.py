import segmentation_models_pytorch as smp
import os
import torch.nn as nn


# model 불러오기
# 출력 label 수 정의 (classes=11)
class UNet(nn.Module):
    def __init__(self, encoder_name="efficientnet-b0", encoder_weights="imagenet", in_channels=3, num_classes=11):
        super(UNet, self).__init__()
        self.model = smp.Unet(
            encoder_name=encoder_name,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=encoder_weights,  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=in_channels,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=num_classes,  # model output channels (number of classes in your dataset)
        )

    def forward(self, x):
        return self.model(x)


class UnetPlusPlus(nn.Module):
    def __init__(self, encoder_name="timm-efficientnet-b0", encoder_weights="imagenet", in_channels=3, num_classes=11):
        super(UnetPlusPlus, self).__init__()
        self.model = smp.UnetPlusPlus(
            encoder_name=encoder_name,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=encoder_weights,  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=in_channels,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=num_classes,  # model output channels (number of classes in your dataset)
        )

    def forward(self, x):
        return self.model(x)


"""
encoder_name
https://github.com/qubvel/segmentation_models.pytorch/tree/35d79c1aa5fb26ba0b2c1ec67084c66d43687220/segmentation_models_pytorch/encoders
encoders.update(resnet_encoders)
resnet18, resnet50, resnext50_32x4d, resnext101_32x4d, resnext101_32x8d, resnext101_32x16d,
resnext101_32x32d, resnext101_32x48d

encoders.update(dpn_encoders)
dpn68, dpn68b, dpn92, dpn98, dpn107, dpn131

encoders.update(vgg_encoders)
vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn

encoders.update(senet_encoders)
senet154, se_resnet50, se_resnet101, se_resnet152, se_resnext50_32x4d, se_resnext101_32x4d

encoders.update(densenet_encoders)
densenet121, densenet169, densenet201, densenet161

encoders.update(inceptionresnetv2_encoders)
inceptionresnetv2

encoders.update(inceptionv4_encoders)
inceptionv4

encoders.update(efficient_net_encoders)
efficientnet-b0, efficientnet-b1, efficientnet-b2, efficientnet-b3,
efficientnet-b4, efficientnet-b5, efficientnet-b6, efficientnet-b7

encoders.update(mobilenet_encoders)
mobilenet_v2

encoders.update(xception_encoders)
xception

encoders.update(timm_efficientnet_encoders)
timm-efficientnet-b0, timm-efficientnet-b1, timm-efficientnet-b3, timm-efficientnet-b4,
timm-efficientnet-b5, timm-efficientnet-b7, timm-efficientnet-b8, timm-efficientnet-12,

encoders.update(timm_resnest_encoders)
timm-resnest14d, timm-resnest26d, timm-resnest50d, timm-resnest101e,
timm-resnest200e, timm-resnest269e, timm-resnest50d_4s2x40d, timm-resnest50d_1s4x24d

encoders.update(timm_res2net_encoders)
timm-res2net50_26w_4s, timm-res2net101_26w_4s, timm-res2net50_26w_6s, timm-res2net50_26w_8s,
timm-res2net50_48w_2s, timm-res2net50_14w_8s, timm-res2next50

encoders.update(timm_regnet_encoders)
timm-regnetx_002, timm-regnetx_004, timm-regnetx_006, timm-regnetx_008, timm-regnetx_016,
timm-regnetx_032, timm-regnetx_040, timm-regnetx_064, timm-regnetx_080, timm-regnetx_120,
timm-regnetx_160, ...

encoders.update(timm_sknet_encoders)
timm-skresnet18, timm-skresnet34, imm-skresnext50_32x4d

encoders.update(timm_mobilenetv3_encoders)
timm-mobilenetv3_large_075, timm-mobilenetv3_large_100, timm-mobilenetv3_large_minimal_100,
timm-mobilenetv3_small_075, timm-mobilenetv3_small_100, timm-mobilenetv3_small_minimal_100

encoders.update(timm_gernet_encoders)
timm-gernet_s, timm-gernet_m, timm-gernet_l

"""
