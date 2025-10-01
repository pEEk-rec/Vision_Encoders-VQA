from mmengine.config import read_base
from mmaction.models.heads.Swin_multitask_head import MultitaskHead  # Change import

with read_base():
    from ..._base_.models.swin_tiny import *  # This now imports MultitaskRecognizer
    from ..._base_.default_runtime import *

# Configure model for multitask learning
model['backbone']['pretrained'] = None
model['backbone']['pretrained2d'] = False

# Update head configuration (since base model now has MultitaskHead)
model['cls_head']['num_classes_action'] = 400  # Kinetics classes
model['cls_head']['num_classes_quality'] = 5   # Quality issue classes
model['cls_head']['in_channels'] = 768
model['cls_head']['dropout_ratio'] = 0.5

# Optimizer configuration with different learning rates
from mmengine.optim import OptimWrapper
from mmaction.engine import SwinOptimWrapperConstructor
from mmengine.runner import EpochBasedTrainLoop, ValLoop, TestLoop

optim_wrapper = dict(
    type=OptimWrapper,
    optimizer=dict(type='torch.optim.AdamW', lr=1e-4, betas=(0.9, 0.999), weight_decay=0.01),  # Reduced base LR
    constructor=SwinOptimWrapperConstructor,
    paramwise_cfg=dict(
        absolute_pos_embed=dict(decay_mult=0.),
        relative_position_bias_table=dict(decay_mult=0.),
        norm=dict(decay_mult=0.),
        backbone=dict(lr_mult=0.1),    # Backbone gets 1e-5
        cls_head=dict(lr_mult=1.0)))   # Heads get 1e-4

# Training configuration
train_cfg = dict(type=EpochBasedTrainLoop, max_epochs=50, val_begin=1, val_interval=5)  # More epochs
val_cfg = dict(type=ValLoop)
test_cfg = dict(type=TestLoop)

# Multitask evaluation metrics
from mmaction.evaluation import AccMetric, MSEMetric

val_evaluator = [
    dict(type=AccMetric, metric_list=['top_k_accuracy'], top_k=(1, 5)),  # Action accuracy
    dict(type=MSEMetric),  # MOS regression
    # Add custom quality classification metric if needed
]
test_evaluator = val_evaluator

# Data pipeline - same as yours
file_client_args = dict(io_backend='disk')

train_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

val_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1, test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

test_pipeline = val_pipeline  # Simplified for multitask

# Data loaders - You'll need a custom dataset class for multitask labels
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        type='MultitaskVideoDataset',  # Custom dataset class needed
        ann_file=r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\tools\data\Konvid15k-b\annotations\train_multitask.txt',  # Updated annotation format
        data_prefix=dict(video=r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\tools\data\Konvid15k-b\videos_train'),
        pipeline=train_pipeline,
    )
)

val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        type='MultitaskVideoDataset',
        ann_file=r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\tools\data\Konvid15k-b\annotations\val_multitask.txt',
        data_prefix=dict(video=r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\tools\data\Konvid15k-b\videos_val'),
        pipeline=val_pipeline,
        test_mode=True,
    )
)

test_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        type='MultitaskVideoDataset',
        ann_file=r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\tools\data\Konvid15k-b\annotations\test_multitask.txt',
        data_prefix=dict(video=r'D:\Academics\SEM-5\NVIDIA_miniproj\mmaction2-1\tools\data\Konvid15k-b\videos_test'),
        pipeline=test_pipeline,
        test_mode=True,
    )
)

# Enable mixed precision
fp16 = dict(loss_scale=512.0)
