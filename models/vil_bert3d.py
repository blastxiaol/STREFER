import torch
import torch.nn as nn
from torchvision.ops import RoIAlign
from .backbone.pointnet import PointNetPP
from .backbone.resnet import ResNet
from .vision_language_bert.vil_bert import ViLBert

class LinearBlock(nn.Module):
    def __init__(self, in_size, out_size, linear_layer_num, dropout=0.):
        super().__init__()
        self.dropout = dropout
        self.layer_num = linear_layer_num

        self.linear= nn.ModuleList()
        for i in range(self.layer_num):
            if i == 0:
                self.linear.append(nn.Linear(in_size, out_size))
            else:
                self.linear.append(nn.Linear(out_size, out_size))
            if i != (self.layer_num - 1):
                self.linear.append(nn.ReLU())
                self.linear.append(nn.Dropout(self.dropout))
            
    def forward(self, x):
        for block in self.linear:
            x = block(x)
        return x

class PointCloudExtactor(nn.Module):
    def __init__(self, args):
        super().__init__()
        dim = 20 if not args.no_rgb else 1
        self.pointnet = PointNetPP(sa_n_points=[32, 16, 16],
                                   sa_n_samples=[32, 32, 32],
                                   sa_radii=[0.2, 0.4, 0.4],
                                   sa_mlps=[[dim, 64, 64, 128],
                                            [128, 128, 128, 256],
                                            [256, 256, 512, args.pc_out_dim]])
        self.proj = nn.Linear(args.pc_out_dim, args.vis_out_dim)


    def forward(self, points):
        point_cloud_feature = self.pointnet.get_siamese_features(points, aggregator=torch.stack)
        point_cloud_feature = self.proj(point_cloud_feature)
        return point_cloud_feature

class ImageExtractor(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.feature_extractor = ResNet(args)
        self.output_dim = self.feature_extractor.output_dim
        self.roi_align = RoIAlign(args.roi_size, spatial_scale=args.spatial_scale, sampling_ratio=-1)
        
        if args.objects_aggregator == 'pool':
            self.aggegator = nn.AdaptiveAvgPool1d(output_size=1)
        elif args.objects_aggregator == 'linear':
            self.aggegator = nn.Linear(args.roi_size[0]*args.roi_size[1], 1)
        else:
            raise NotImplementedError("Wrong objects aggregator")
        
        self.proj = nn.Linear(self.output_dim, args.vis_out_dim)


    def extract_box_feature(self, image_feature, boxes):
        object_boxes = []
        B, N, _ = boxes.shape
        for i in range(B):
            object_boxes.append(boxes[i])
        objects_feature = self.roi_align(image_feature, object_boxes)
        objects_feature = objects_feature.view(objects_feature.shape[0], objects_feature.shape[1], -1)
        objects_feature = self.aggegator(objects_feature).view(B, N, self.output_dim)
        return objects_feature
    
    def forward(self, image, boxes):
        image_feature = self.feature_extractor(image)
        objects_feature = self.extract_box_feature(image_feature, boxes)
        objects_feature = self.proj(objects_feature)
        return objects_feature

class PointcloudImageFusion(nn.Module):
    def __init__(self, args, image_out_dim):
        super().__init__()
        self.linear = nn.Sequential(
                LinearBlock(args.vis_out_dim * 2, args.vis_out_dim, args.linear_layer_num, args.dropout),
                nn.LayerNorm(args.vis_out_dim)
            )
    
    def forward(self, image_feature, point_cloud_feature):
        visual_feature = torch.cat([image_feature, point_cloud_feature], dim=2)
        visual_feature = self.linear(visual_feature)
        return visual_feature

class ViLBert3D(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.point_cloud_extractor = PointCloudExtactor(args)
        self.image_extractor = ImageExtractor(args)
        self.fusion = PointcloudImageFusion(args, self.image_extractor.output_dim)
        spatial_dim = 8
        if args.use_view:
            spatial_dim = 7
        if args.use_vel:
            spatial_dim = 9
        self.matching = ViLBert(spatial_dim, args.vilbert_config_path, args.vil_pretrained_file)
    
    def forward(self, image, boxes2d, points, spatial, vis_mask, token, mask, segment_ids):
        # extract point cloud feature
        point_cloud_feature = self.point_cloud_extractor(points)

        # extract image feature
        image_feature = self.image_extractor(image, boxes2d)

        # fusion
        visual_feature = self.fusion(image_feature, point_cloud_feature)

        # matching
        score = self.matching(token, visual_feature, spatial, segment_ids, mask, vis_mask)
        return score
