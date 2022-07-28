class_names = ['person']
input_modality = dict(use_lidar=True, use_camera=False)
train_pipeline = [
    dict(
        type='LoadPointsFromMultiFrame',
        load_dim=4,
        coord_type='LIDAR',
        use_dim=range(0, 4)),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='PointsRangeFilter',
        point_cloud_range=[0, -20.48, -4, 30.72, 20.48, 1]),
    dict(
        type='ObjectRangeFilter',
        point_cloud_range=[0, -20.48, -4, 30.72, 20.48, 1]),
    dict(type='DefaultFormatBundle3D', class_names=['person']),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromMultiFrame', coord_type='LIDAR', load_dim=4,
        use_dim=4),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='PointsRangeFilter',
                point_cloud_range=[0, -20.48, -4, 30.72, 20.48, 1]),
            dict(
                type='DefaultFormatBundle3D',
                class_names=['person'],
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]
eval_pipeline = [
    dict(
        type='LoadPointsFromMultiFrame', coord_type='LIDAR', load_dim=4,
        use_dim=4),
    dict(
        type='DefaultFormatBundle3D', class_names=['person'],
        with_label=False),
    dict(type='Collect3D', keys=['points'])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='SHtechDataset',
        data_root='/remote-home/share/SHTperson/',
        ann_file='/remote-home/share/SHTperson/save_info/sht_infos_train19.pkl',
        pipeline=[
            dict(
                type='LoadPointsFromMultiFrame',
                coord_type='LIDAR',
                load_dim=4,
                use_dim=4),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True),
            dict(
                type='PointsRangeFilter',
                point_cloud_range=[0, -20.48, -4, 30.72, 20.48, 1]),
            dict(
                type='ObjectRangeFilter',
                point_cloud_range=[0, -20.48, -4, 30.72, 20.48, 1]),
            dict(type='DefaultFormatBundle3D', class_names=['person']),
            dict(
                type='Collect3D',
                keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
        ],
        modality=dict(use_lidar=True, use_camera=False),
        classes=['person'],
        test_mode=False),
    val=dict(
        type='SHtechDataset',
        data_root='/remote-home/share/SHTperson/',
        ann_file=
        '/remote-home/share/SHTperson/save_info/sht_infos_val_2345678.pkl',
        pipeline=[
            dict(
                type='LoadPointsFromMultiFrame',
                coord_type='LIDAR',
                load_dim=4,
                use_dim=4),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='PointsRangeFilter',
                        point_cloud_range=[0, -20.48, -4, 30.72, 20.48, 1]),
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=['person'],
                        with_label=False),
                    dict(type='Collect3D', keys=['points'])
                ])
        ],
        modality=dict(use_lidar=True, use_camera=False),
        classes=['person'],
        test_mode=True),
    test=dict(
        type='SHtechDataset',
        data_root='/remote-home/share/SHTperson/',
        ann_file=
        # '/remote-home/share/SHTperson/save_info/sht_infos_train19.pkl',
        '/remote-home/share/SHTperson/save_info/sht_infos_val_2345678.pkl',
        # "/remote-home/share/SHTperson/save_info/sht_infos_test_2345678.pkl",
        pipeline=[
            dict(
                type='LoadPointsFromMultiFrame',
                coord_type='LIDAR',
                load_dim=4,
                use_dim=4),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='PointsRangeFilter',
                        point_cloud_range=[0, -20.48, -4, 30.72, 20.48, 1]),
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=['person'],
                        with_label=False),
                    dict(type='Collect3D', keys=['points'])
                ])
        ],
        modality=dict(use_lidar=True, use_camera=False),
        classes=['person'],
        test_mode=True))
point_cloud_range = [0, -20.48, -4, 30.72, 20.48, 1]
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
voxel_size = [0.12, 0.16, 0.2]
model = dict(
    type='CenterPoint',
    pts_voxel_layer=dict(
        point_cloud_range=[0, -20.48, -4, 30.72, 20.48, 1],
        max_num_points=100,
        voxel_size=[0.12, 0.16, 0.2],
        max_voxels=(90000, 120000)),
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
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        out_channels=[256, 256],
        upsample_strides=[2, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    pts_bbox_head=dict(
        type='CenterHead',
        in_channels=512,
        tasks=[dict(num_class=1, class_names=['person'])],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
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
        norm_bbox=True),
    train_cfg=dict(
        pts=dict(
            point_cloud_range=[0, -20.48, -4, 30.72, 20.48, 1],
            grid_size=[256, 256, 25],
            voxel_size=[0.12, 0.16, 0.2],
            out_size_factor=4,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.4, 0.4])),
    test_cfg=dict(
        pts=dict(
            post_center_limit_range=[0, -20.0, -3, 25.0, 20.0, -1],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[0.2],
            score_threshold=0.2,
            out_size_factor=4,
            voxel_size=[0.12, 0.16],
            nms_type='circle',
            pre_max_size=1000,
            post_max_size=100,
            nms_thr=0.2)))
checkpoint_config = dict(interval=1)
evaluation = dict(interval=3)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = '../tracking_models/1-21-seg-bb'
# load_from = '../painting_models/voxel_train19_seg_feature_75epoch.pth'
resume_from = None
workflow = [('train', 1)]
gpu_ids = range(0, 1)