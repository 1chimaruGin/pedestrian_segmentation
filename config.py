import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

class Config(object):
    
    BACKBONE = torchvision.models.mobilenet_v2(pretrained=True).features

    BACKBONE.out_channels = 1280

    ANCHOR_GENERATOR = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                   aspect_ratios=((0.5, 1.0, 2.0),))

    ROI_POOLER = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
                                                output_size=7,
                                                sampling_ratio=2)

    MODEL = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
                                                output_size=7,
                                                sampling_ratio=2)

    def __init__(self):
        pass
    

