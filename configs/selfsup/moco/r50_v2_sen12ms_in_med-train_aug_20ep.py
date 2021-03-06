_base_ = '../../base.py'
# model settings
model = dict(
    type='MOCO',
    queue_len=65536,
    feat_dim=128,
    momentum=0.999,
    backbone=dict(
        type='ResNet',
        depth=50,
        in_channels=3,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='BN')),
    neck=dict(
        type='NonLinearNeckV1',
        in_channels=2048,
        hid_channels=2048,
        out_channels=128,
        with_avg_pool=True),
    head=dict(type='ContrastiveHead', temperature=0.2),
    input_module_q=dict(
        type='Conv1x1Block',
        in_channels=10,
        out_channels=3,
    ),
    input_module_k=dict(
        type='Conv1x1Block',
        in_channels=2,
        out_channels=3,
    ),
)
# dataset settings
data_source_cfg = dict(
    type='Sen12MS',
    memcached=False,
    mclient_path='/mnt/lustre/share/memcached_client')
dataset_name = "35k_samples"
data_train_list = f'data/sen12ms/meta/{dataset_name}.txt'
data_train_root = 'data/sen12ms/data'
dataset_type = 'ContrastiveMSDataset'
# img_norm_cfg = dict(mean=[0.368, 0.381, 0.3436], std=[0.2035, 0.1854, 0.1849])
# img_norm_cfg = dict(s1_mean=[-11.76858, -18.294598],
#                     s2_mean=[1226.4215, 1137.3799, 1139.6792, 1350.9973, 1932.9058, 2211.1584, 2154.9846, 2409.1128,
#                              2001.8622, 1356.0801],
#                     s1_std=[4.525339, 4.3586307],
#                     s2_std=[741.6254, 740.883, 960.1045, 946.76056, 985.52747, 1082.4341, 1057.7628, 1136.1942,
#                             1132.7898, 991.48016])

# statistics for "multi_label"
img_norm_cfg = dict(bands_mean={'s1_mean': [-11.76858, -18.294598],
                                's2_mean': [1226.4215, 1137.3799, 1139.6792, 1350.9973, 1932.9058,
                                            2211.1584, 2154.9846, 2409.1128, 2001.8622, 1356.0801]},
                    bands_std={'s1_std': [4.525339, 4.3586307],
                               's2_std': [741.6254, 740.883, 960.1045, 946.76056, 985.52747,
                                          1082.4341, 1057.7628, 1136.1942, 1132.7898, 991.48016]})

train_pipeline = [
    # dict(type='RandomResizedCrop', size=224, scale=(0.2, 1.)),
    # dict(
    #     type='RandomAppliedTrans',
    #     transforms=[
    #         dict(
    #             type='ColorJitter',
    #             brightness=0.4,
    #             contrast=0.4,
    #             saturation=0.4,
    #             hue=0.4)
    #     ],
    #     p=0.8),
    # dict(type='RandomGrayscale', p=0.2),
    # dict(
    #     type='RandomAppliedTrans',
    #     transforms=[
    #         dict(
    #             type='GaussianBlur',
    #             sigma_min=0.1,
    #             sigma_max=2.0,
    #         )
    #     ],
    #     p=0.5),
    # dict(type='RandomHorizontalFlip'),
    # TODO: Fix the following
    # dict(type='Alb_RandomCrop'),
    # dict(type='Alb_ColorJitter'),

    dict(type='Alb_GaussianBlur'),
    dict(type='Alb_ElasticTransform'),
    dict(type='Alb_Blur'),
    dict(type='Alb_VerticalFlip'),
    dict(type='Alb_HorizontalFlip'),
    dict(type='Alb_RandomBrightnessContrast'),
    dict(type='Sen12msToTensor'),
    dict(type='Sen12msNormalize', **img_norm_cfg),
]
data = dict(
    imgs_per_gpu=64,  # total 64*4=256
    workers_per_gpu=4,
    drop_last=True,
    train=dict(
        type=dataset_type,
        data_source=dict(
            list_file=data_train_list, root=data_train_root,
            name=dataset_name,
            **data_source_cfg),
        pipeline=train_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.03, weight_decay=0.0001, momentum=0.9)
# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0.)
checkpoint_config = dict(interval=20)
# runtime settings
total_epochs = 20
