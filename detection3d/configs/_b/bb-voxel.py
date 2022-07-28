point_cloud_range=[0, -20.48, -4, 30.72, 20.48, 1]
optimizer = dict(type='AdamW', lr=0.0002, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='cyclic',
    target_ratio=(10, 0.0001),
    cyclic_times=1,
    step_ratio_up=0.4)
momentum_config = dict(
    policy='cyclic',
    target_ratio=(0.8947368421052632, 1),
    cyclic_times=1,
    step_ratio_up=0.4)
runner = dict(type='EpochBasedRunner', max_epochs=150)
point_cloud_range = [0, -20.48, -4, 30.72, 20.48, 1]
voxel_size = [0.12, 0.16, 0.2]
model = dict(
    type='CenterPoint',
    pts_voxel_layer=dict(
        point_cloud_range=[0, -20.48, -4, 30.72, 20.48, 1],
        max_num_points=100, voxel_size=voxel_size, max_voxels=(90000, 120000)),
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=4),
    pts_middle_encoder=dict(
        type='SparseEncoder',
        in_channels=4,
        sparse_shape=[41, 256, 256],
        output_channels=128,
        order=('conv', 'norm', 'act'),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128,
                                                                      128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type='basicblock'),
    pts_backbone=dict(
        type='SECOND',
        in_channels=256,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 1],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        out_channels=[256, 256],
        upsample_strides=[2, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    pts_bbox_head=dict(
        type='MultiCenterHead',
        in_channels=sum([256, 256]),
        tasks=[dict(num_class=1, class_names=['person'])],
        common_heads=dict(reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2),vel=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            pc_range=[0, -20.48],
            type='CenterPointBBoxCoder',
            post_center_range=[0, -20.0, -2, 30.0, 20.0, 0],
            max_num=500,
            score_threshold=0.2,
            out_size_factor=4,
            voxel_size=[0.12, 0.16],
            code_size=9),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        # loss_diff_frame = dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            point_cloud_range=[0, -20.48, -4, 30.72, 20.48, 1],
            grid_size=[256, 256, 25],
            voxel_size=voxel_size,
            out_size_factor=4,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,0.4,0.4])),
    test_cfg=dict(
        pts=dict(
            post_center_limit_range=[0, -20.0, -3, 25.0, 20.0, -1],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[0.3],
            score_threshold=0.2,
            out_size_factor=4,
            voxel_size=voxel_size[:2],
            nms_type='circle',
            pre_max_size=1000,
            post_max_size=100,
            nms_thr=0.2)))
