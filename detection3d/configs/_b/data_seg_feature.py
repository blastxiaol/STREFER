dataset_type = 'SHtechDataset'
data_root = '/remote-home/share/SHTperson/'
class_names = ['person']
input_modality = dict(use_lidar=True, use_camera=False)
train_pipeline = [
    dict(type='LoadPointsFromMultiFrame',
                load_dim=23,
                coord_type='LIDAR',
                use_dim=range(23)),
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
    dict(type='LoadPointsFromMultiFrame', coord_type='LIDAR', load_dim=23, use_dim=23),
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
    dict(type='LoadPointsFromMultiFrame', coord_type='LIDAR', load_dim=23, use_dim=23),
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
        pipeline=train_pipeline,
        modality=dict(use_lidar=True, use_camera=False),
        classes=['person'],
        test_mode=False),
    val=dict(
        type='SHtechDataset',
        data_root='/remote-home/share/SHTperson/',
        ann_file='/remote-home/share/SHTperson/save_info/sht_infos_val_2345678.pkl',
        pipeline=test_pipeline,
        modality=dict(use_lidar=True, use_camera=False),
        classes=['person'],
        test_mode=True),
    test=dict(
        type='SHtechDataset',
        data_root='/remote-home/share/SHTperson/',
        ann_file=
        "/remote-home/share/SHTperson/save_info/sht_infos_test_2345678.pkl",
        pipeline=test_pipeline,
        modality=dict(use_lidar=True, use_camera=False),
        classes=['person'],
        test_mode=True))