_base_ = [
    './data_seg_feature.py',
    './bb-voxel.py'
]
point_cloud_range=[0, -20.48, -4, 30.72, 20.48, 1]
model = dict(
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=23),
    pts_middle_encoder=dict(
        type='SparseEncoder',
        in_channels=23,
        sparse_shape=[41, 256, 256],
        output_channels=128,
        order=('conv', 'norm', 'act'),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128,
                                                                      128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type='basicblock')
)
checkpoint_config = dict(interval=5)
evaluation = dict(interval=5)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'work_dirs/tracking_models_seg_23'
load_from = 'pths/voxel_sweep.pth'                # '../painting_models/voxel_train19_seg_feature_75epoch.pth'
resume_from = None
workflow = [('train', 1)]
# gpu_ids = range(0)
runner = dict(type='EpochBasedRunner', max_epochs=150)
find_unused_parameters = True
