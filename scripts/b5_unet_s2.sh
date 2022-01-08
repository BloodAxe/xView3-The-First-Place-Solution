export XVIEW3_DATA_DIR="/home/bloodaxe/data/xview3"
export OMP_NUM_THREADS=16

torchrun --standalone --nnodes=1 --nproc_per_node=2 train_multilabel.py\
    model=centernet/b5_unet_s2\
    loss=centernet/circlenet_loss\
    train=large train.show=False train.early_stopping=15\
    augs=flips_light_medium\
    optimizer=adamw scheduler=cos6\
    dataset=full_4fold\
    dataset.ignore_low_confidence_labels=True dataset.ignore_low_confidence_detections=True\
    dataset.fold=1\
    sampler=balanced sampler.num_samples=16536\
    train.loaders.train.batch_size=6\
    train.loaders.train.num_workers=6\
    train.loaders.valid.batch_size=4\
    train.loaders.valid.num_workers=4\
    torch.channels_last=True checkpoint=/home/bloodaxe/develop/xView3/runs/211126_13_52_b5_unet_s2_full_4fold_circlenet_loss_flips_light_medium_fold1/checkpoints_metrics_aggregate/last.pth

torchrun --standalone --nnodes=1 --nproc_per_node=2 train_multilabel.py\
    model=centernet/b5_unet_s2\
    loss=centernet/circlenet_loss\
    train=large train.show=False train.early_stopping=15\
    augs=flips_light_medium\
    optimizer=adamw scheduler=cos6\
    dataset=full_4fold\
    dataset.ignore_low_confidence_labels=True dataset.ignore_low_confidence_detections=True\
    dataset.fold=2\
    sampler=balanced sampler.num_samples=16536\
    train.loaders.train.batch_size=6\
    train.loaders.train.num_workers=6\
    train.loaders.valid.batch_size=4\
    train.loaders.valid.num_workers=4\
    torch.channels_last=True checkpoint=/home/bloodaxe/develop/xView3/runs/211126_15_31_b5_unet_s2_full_4fold_circlenet_loss_flips_light_medium_fold2/checkpoints_metrics_aggregate/last.pth

torchrun --standalone --nnodes=1 --nproc_per_node=2 train_multilabel.py\
    model=centernet/b5_unet_s2\
    loss=centernet/circlenet_loss\
    train=large train.show=False train.early_stopping=15\
    augs=flips_light_medium\
    optimizer=adamw scheduler=cos6\
    dataset=full_4fold\
    dataset.ignore_low_confidence_labels=True dataset.ignore_low_confidence_detections=True\
    dataset.fold=3\
    sampler=balanced sampler.num_samples=16536\
    train.loaders.train.batch_size=6\
    train.loaders.train.num_workers=6\
    train.loaders.valid.batch_size=4\
    train.loaders.valid.num_workers=4\
    torch.channels_last=True
