from soupsieve import match
import torch
import torch.nn as nn
from torchvision.ops import RoIAlign
from .backbone.pointnet import PointNetPP
from .backbone.resnet import ResNet
from .vision_language_bert.vil_bert import ViLBert

def matching(feature, prev_feature, rel_dist_mask, threshold, prev_spatial=None):
    B = rel_dist_mask.shape[0]
    N = rel_dist_mask.shape[2]
    ordered_prev_feature = torch.zeros_like(feature)
    visited = torch.zeros((B, N), dtype=torch.bool, device='cpu')
    new_prev_spatial = None
    if prev_spatial is not None:
        new_prev_spatial = torch.zeros_like(prev_spatial)
    for b in range(B):
        for i in range(rel_dist_mask.shape[1]):
            main_feat = feature[b, i]       # 512
            max_similarity = -1
            max_idx = -1
            # print(main_feat.shape)
            for j in range(rel_dist_mask.shape[2]):
                if not rel_dist_mask[b, i, j] or visited[b, j]:
                    continue
                prev_feat = prev_feature[b, j]
                similarity = torch.cosine_similarity(main_feat, prev_feat, dim=0).item()
                if similarity > max_similarity and similarity >= threshold:
                    max_similarity = similarity
                    max_idx = j
            if max_idx > -1:
                visited[b, max_idx] = 1
                ordered_prev_feature[b, i] = prev_feature[b, max_idx]
                if new_prev_spatial is not None:
                    new_prev_spatial[b, i] = prev_spatial[b, max_idx]
    return ordered_prev_feature, new_prev_spatial

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


    def forward(self, points):
        point_cloud_feature = self.pointnet.get_siamese_features(points, aggregator=torch.stack)
        # point_cloud_feature = self.proj(point_cloud_feature)
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
        # objects_feature = self.proj(objects_feature)
        return objects_feature

class PointcloudImageFusion(nn.Module):
    def __init__(self, args, image_out_dim):
        super().__init__()
        self.matching_threshold = args.matching_threshold
        self.image_out_dim = image_out_dim

        self.pc_fusion = nn.Sequential(
            LinearBlock(args.pc_out_dim * 2, args.pc_out_dim, args.linear_layer_num, args.dropout),
            nn.Linear(args.pc_out_dim, args.vis_out_dim),
            nn.LayerNorm(args.vis_out_dim)
        )
        self.img_fusion = nn.Sequential(
            LinearBlock(self.image_out_dim * 2, self.image_out_dim, args.linear_layer_num, args.dropout),
            nn.Linear(self.image_out_dim, args.vis_out_dim),
            nn.LayerNorm(args.vis_out_dim)
        )
        self.linear = nn.Sequential(
                LinearBlock(args.vis_out_dim * 2, args.vis_out_dim, args.linear_layer_num, args.dropout),
                nn.LayerNorm(args.vis_out_dim)
        )
    
    def matching_previous_objects(self, image_feature, point_cloud_feature, prev_image_feature, prev_point_cloud_feature, rel_dist_mask, prev_spatial):
        device = image_feature.device
        visual_feature = torch.cat([image_feature, point_cloud_feature], dim=2).cpu()
        prev_visual_feature = torch.cat([prev_image_feature, prev_point_cloud_feature], dim=2).cpu()

        ordered_prev_feature, prev_spatial = matching(visual_feature, prev_visual_feature, rel_dist_mask, self.matching_threshold, prev_spatial)
        ordered_prev_image_feature = ordered_prev_feature[:, :, :self.image_out_dim].to(device)
        ordered_prev_point_cloud_feature = ordered_prev_feature[:, :, self.image_out_dim:].to(device)
        return ordered_prev_image_feature, ordered_prev_point_cloud_feature, prev_spatial

    def forward(self, image_feature, point_cloud_feature, prev_image_feature, prev_point_cloud_feature, rel_dist_mask, prev_spatial=None):
        prev_image_feature, prev_point_cloud_feature, prev_spatial = \
            self.matching_previous_objects(image_feature, point_cloud_feature, prev_image_feature, prev_point_cloud_feature, rel_dist_mask, prev_spatial)

        image_feature = torch.cat([image_feature, prev_image_feature], dim=2)
        image_feature = self.img_fusion(image_feature)
        point_cloud_feature = torch.cat([point_cloud_feature, prev_point_cloud_feature], dim=2)
        point_cloud_feature = self.pc_fusion(point_cloud_feature)
        
        visual_feature = torch.cat([image_feature, point_cloud_feature], dim=2)
        visual_feature = self.linear(visual_feature)
        return visual_feature, prev_spatial

class ViLBert3DMF(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.point_cloud_extractor = PointCloudExtactor(args)
        self.image_extractor = ImageExtractor(args)
        self.fusion = PointcloudImageFusion(args, self.image_extractor.output_dim)
        factor = 1
        self.cat_spatial = args.cat_spatial
        if self.cat_spatial:
            factor = 2
        spatial_dim = 8
        if args.use_view:
            spatial_dim = 7
        if args.use_vel:
            spatial_dim = 9
        spatial_dim = spatial_dim * factor
        self.matching = ViLBert(spatial_dim, args.vilbert_config_path, args.vil_pretrained_file)

    
    def forward(self, image, boxes2d, points, spatial, vis_mask, 
                    prev_image, prev_boxes2d, prev_points, prev_spatial, prev_vis_mask, 
                    token, mask, segment_ids, rel_dist_mask, obj_center, **kwargs):
        """
        image:         [B, 3, W, H]            | prev_image:    [B, 3, W, H] 
        boxes2d:       [B, N_obj, 4]           | prev_boxes2d:  [B, N_obj, 4]
        points:        [B, N_obj, Np, 3+C]     | prev_points:   [B, N_obj, Np, 3+C]
        spatial:       [B, N_obj, dim]         | prev_spatial:  [B, N_obj, dim]
        vis_mask:      [B, N_obj]              | prev_vis_mask: [B, N_obj]
        token:         [B, seq_len]
        mask:          [B, seq_len]
        segment_ids:   [B, seq_len]
        rel_dist_mask: [B, N_obj, N_obj]
        """
        # extract point cloud feature
        point_cloud_feature = self.point_cloud_extractor(points)
        prev_point_cloud_feature = self.point_cloud_extractor(prev_points)
        
        # extract image feature
        image_feature = self.image_extractor(image, boxes2d)
        prev_image_feature = self.image_extractor(prev_image, prev_boxes2d)

        # fusion
        visual_feature, prev_spatial = self.fusion(image_feature, point_cloud_feature, prev_image_feature, prev_point_cloud_feature, rel_dist_mask, prev_spatial)
        if self.cat_spatial:        
            spatial = torch.cat([spatial, prev_spatial], dim=2)
        # matching
        score = self.matching(token, visual_feature, spatial, segment_ids, mask, vis_mask)
        return score
