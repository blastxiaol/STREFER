import torch.nn as nn
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152

class ResNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        resnet_layer_num = args.resnet_layer_num
        if resnet_layer_num == 18:
            resnet = resnet18
            self.output_dim = 512
        elif resnet_layer_num == 34:
            resnet = resnet34
            self.output_dim = 512
        elif resnet_layer_num == 50:
            resnet = resnet50
            self.output_dim = 2048
        elif resnet_layer_num == 101:
            resnet = resnet101
            self.output_dim = 2048
        elif resnet_layer_num == 152:
            resnet = resnet152
            self.output_dim = 2048
        else:
            raise ValueError(f"args.resnet_layer_num shoud in [18, 34, 50, 101, 152], not {resnet_layer_num}")
        
        pretrained = not args.no_pretrained_resnet
        self.resnet = nn.Sequential(*list(resnet(pretrained=pretrained).children())[:-2])
    
    def forward(self, x):
        return self.resnet(x)