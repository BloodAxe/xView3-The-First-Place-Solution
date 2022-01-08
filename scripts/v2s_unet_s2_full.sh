export XVIEW3_DATA_DIR="/home/bloodaxe/data/xview3"
export OMP_NUM_THREADS=16

torchrun --standalone --nnodes=1 --nproc_per_node=2 train_multilabel.py\
    model=centernet/v2s_unet_s2\
    loss=centernet/rfl_soft_bce_smooth_l1_regularized\
    train=onek train.show=False train.early_stopping=50\
    augs=flips_light_medium\
    ema=ema_batch\
    optimizer=radam\
    scheduler=cos6\
    dataset=full_4fold\
    dataset.ignore_low_confidence_labels=True dataset.ignore_low_confidence_detections=True\
    dataset.fold=0\
    sampler=balanced sampler.num_samples=8192\
    train.loaders.train.batch_size=6\
    train.loaders.train.num_workers=8\
    train.loaders.valid.batch_size=6\
    train.loaders.valid.num_workers=4\
    torch.channels_last=True seed=111\
    checkpoint=/home/bloodaxe/develop/xView3/runs/211121_02_06_v2s_unet_s2_full_4fold_rfl_soft_bce_mse_regularized_flips_light_medium_fold0/checkpoints_metrics_aggregate/best.pth

torchrun --standalone --nnodes=1 --nproc_per_node=2 train_multilabel.py\
    model=centernet/v2s_unet_s2\
    loss=centernet/rfl_soft_bce_smooth_l1_regularized\
    train=onek train.show=False train.early_stopping=50\
    augs=flips_light_medium\
    optimizer=radam\
    ema=ema_batch\
    scheduler=cos6\
    dataset=full_4fold\
    dataset.ignore_low_confidence_labels=True dataset.ignore_low_confidence_detections=True\
    dataset.fold=1\
    sampler=balanced sampler.num_samples=8192\
    train.loaders.train.batch_size=6\
    train.loaders.train.num_workers=8\
    train.loaders.valid.batch_size=6\
    train.loaders.valid.num_workers=4\
    torch.channels_last=True seed=222\
    checkpoint=runs/211119_11_35_v2s_unet_s2_leaky_valid_4fold_rfl_soft_bce_mse_regularized_flips_light_medium_fold1/checkpoints_metrics_aggregate/best.pth


torchrun --standalone --nnodes=1 --nproc_per_node=2 train_multilabel.py\
    model=centernet/v2s_unet_s2\
    loss=centernet/rfl_soft_bce_smooth_l1_regularized\
    train=onek train.show=False train.early_stopping=50\
    augs=flips_light_medium\
    optimizer=radam\
    ema=ema_batch\
    scheduler=cos6\
    dataset=full_4fold\
    dataset.ignore_low_confidence_labels=True dataset.ignore_low_confidence_detections=True\
    dataset.fold=2\
    sampler=balanced sampler.num_samples=8192\
    train.loaders.train.batch_size=6\
    train.loaders.train.num_workers=8\
    train.loaders.valid.batch_size=6\
    train.loaders.valid.num_workers=4\
    torch.channels_last=True seed=333\
    checkpoint=runs/211119_21_11_v2s_unet_s2_leaky_valid_4fold_rfl_soft_bce_mse_regularized_flips_light_medium_fold2/checkpoints_metrics_aggregate/best.pth

torchrun --standalone --nnodes=1 --nproc_per_node=2 train_multilabel.py\
    model=centernet/v2s_unet_s2\
    loss=centernet/rfl_soft_bce_smooth_l1_regularized\
    train=onek train.show=False train.early_stopping=50\
    augs=flips_light_medium\
    ema=ema_batch\
    optimizer=radam\
    scheduler=cos6\
    dataset=full_4fold\
    dataset.ignore_low_confidence_labels=True dataset.ignore_low_confidence_detections=True\
    dataset.fold=3\
    sampler=balanced sampler.num_samples=8192\
    train.loaders.train.batch_size=6\
    train.loaders.train.num_workers=8\
    train.loaders.valid.batch_size=6\
    train.loaders.valid.num_workers=4\
    torch.channels_last=True seed=444\
    checkpoint=runs/211120_01_38_v2s_unet_s2_leaky_valid_4fold_rfl_soft_bce_mse_regularized_flips_light_medium_fold3/checkpoints_metrics_aggregate/best.pth
