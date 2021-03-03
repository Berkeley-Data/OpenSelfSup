_base_ = "finetune-eval-base.py"

# dataset settings
data_source_cfg = dict(
    type="ImageNet",
    memcached=False,
    mclient_path='/no/matter',
    # this will be ignored if type != ImageListMultihead
    
)


data_train_list = "data/domain_net_clipart/meta/train_labeled-1000.txt"
data_train_root = 'data/domain_net_clipart'

data_val_list = "data/domain_net_clipart/meta/val_labeled.txt"
data_val_root = 'data/domain_net_clipart'

data_test_list = "data/domain_net_clipart/meta/test_labeled.txt"
data_test_root = 'data/domain_net_clipart'

dataset_type = "ClassificationDataset"
img_norm_cfg = dict(mean=[0.7362, 0.7167, 0.6834], std=[0.3642, 0.3654, 0.3887])

train_pipeline = [
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomHorizontalFlip'),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
]
test_pipeline = [
    dict(type='Resize', size=256),
    dict(type='CenterCrop', size=224),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
]
data = dict(
    batch_size=64,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_source=dict(
            list_file=data_train_list, root=data_train_root,
            **data_source_cfg),
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_source=dict(
            list_file=data_val_list, root=data_val_root, **data_source_cfg),
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_source=dict(
            list_file=data_test_list, root=data_test_root, **data_source_cfg),
        pipeline=test_pipeline))


custom_hooks = [
    dict(
        name="val",
        type='ValidateHook',
        dataset=data['val'],
        by_epoch=True,
        initial=False,
        interval=1,
        imgs_per_gpu=32,
        workers_per_gpu=4,
        eval_param=dict(topk=(1,5))),
    dict(
        name="test",
        type='ValidateHook',
        dataset=data['test'],
        by_epoch=True,
        initial=False,
        interval=1,
        imgs_per_gpu=32,
        workers_per_gpu=4,
        eval_param=dict(topk=(1,5))),
]

by_iter =False

# learning policy
lr_config = dict(
    by_epoch=True,
    policy='step',
    step=[30,60],
    gamma=0.1  # multiply LR by this number at each step
)

# momentum and weight decay from VTAB and IDRL
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.,
                 paramwise_options={'\Ahead.': dict(lr_mult=100)})


# runtime settings
# total iters or total epochs
total_epochs=90
checkpoint_config = dict(interval=90)

log_config = dict(
    interval=1,
    by_epoch=True,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=True),
        dict(type='TensorboardLoggerHook', by_epoch=True)
    ])
optimizer_config = dict(update_interval=4)
