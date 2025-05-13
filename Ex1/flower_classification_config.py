_base_ = [
    '/content/mmpretrain/configs/_base_/models/resnet50.py',
    '/content/mmpretrain/configs/_base_/datasets/imagenet_bs32.py',
    '/content/mmpretrain/configs/_base_/schedules/imagenet_bs256.py',
    '/content/mmpretrain/configs/_base_/default_runtime.py'
]

# 1. 模型 & 预训练权重：指定模型的分类头 num_classes=5 和 top-k 准确率计算方式，并加载 ImageNet 预训练权重。
model = dict(
    head=dict(num_classes=5, topk=(1,)),
    # 模型初始化时从指定路径URL加载预训练权重，加快收敛并提升性能。
    backbone=dict(
        frozen_stages=3,  # 冻结主干网络的前3层
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth',
            prefix='backbone',
        )
    )
)

# 2. 数据预处理：定义全局的图像归一化参数及类别数。
data_preprocessor = dict(
    mean=[123.675, 116.28, 103.53],  # 图像标准化参数，对应 ImageNet 训练时使用的均值与标准差，以保证输入分布一致。
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
    num_classes=5,
)

# 3. 数据集设置
dataset_type = 'ImageNet'
data_root = '/content/flower_dataset'
classes = [c.strip() for c in open(f'{data_root}/classes.txt')]

# 配置数据加载器，包括批大小、并行进程数、持久化 worker，以及数据前处理流水线。
train_dataloader = dict(
    batch_size=32,
    num_workers=2,
    persistent_workers=True,
    dataset=dict(
        type=dataset_type,
        data_prefix=f'{data_root}/train',
        ann_file=f'{data_root}/train.txt',
        classes=classes,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='RandomResizedCrop', scale=224),
            dict(type='RandomFlip', prob=0.5, direction='horizontal'),
            dict(type='PackInputs')
        ]
    )
)

val_dataloader = dict(
    batch_size=32,
    num_workers=2,
    persistent_workers=True,
    dataset=dict(
        type=dataset_type,
        data_prefix=f'{data_root}/val',
        ann_file=f'{data_root}/val.txt',
        classes=classes,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='ResizeEdge', scale=256, edge='short'),
            dict(type='CenterCrop', crop_size=224),
            dict(type='PackInputs')
        ]
    )
)

val_cfg=dict()
# 确保验证流程完整，指定验证评价指标为 top1 准确率。
val_evaluator = dict(type='Accuracy', topk=(1,))

# 4. 优化器 & 学习率
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=0.001,
        weight_decay=1e-4,
        betas=(0.9, 0.999),
        _delete_=True  # 删除继承的优化器参数
    )
)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=True, begin=0, end=5),  # 学习率热身阶段
    dict(type='ExponentialLR', gamma=0.9, by_epoch=True)  # 指数衰减
]
auto_scale_lr = dict(base_batch_size=256)

# 5. 训练策略（含验证间隔）：指定按 epoch 训练、最大 20 个 epoch、以及每个 epoch 结束后都执行一次验证。
train_cfg = dict(
    by_epoch=True,
    max_epochs=20,
    val_interval=1,  # 每 epoch 验证一次
)


# 6. 测试配置
test_dataloader = dict(
    batch_size=32,
    num_workers=2,
    persistent_workers=True,
    dataset=dict(
        type=dataset_type,
        data_prefix=f'{data_root}/val',  # 测试集路径
        ann_file=f'{data_root}/val.txt',  # 测试集标注文件
        classes=classes,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='ResizeEdge', scale=256, edge='short'),
            dict(type='CenterCrop', crop_size=224),
            dict(type='PackInputs')
        ]
    )
)

test_cfg = dict()
test_evaluator = dict(type='Accuracy', topk=(1,))  # 测试评价指标