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
