export XVIEW3_DATA_DIR="/home/bloodaxe/data/xview3"
export OMP_NUM_THREADS=16

for FOLD in 0 2 3
do
torchrun --standalone --nnodes=1 --nproc_per_node=2 train_multilabel.py\
    model=centernet/b4_unet_s2\
    loss=centernet/rfl_soft_bce_mse_regularized\
    train=large train.show=False train.early_stopping=15\
    augs=flips_light_medium\
    optimizer=adabelief\
    dataset=leaky_valid_4fold\
    dataset.ignore_low_confidence_labels=True dataset.ignore_low_confidence_detections=True\
    dataset.fold=$FOLD\
    sampler=balanced sampler.num_samples=16536\
    train.loaders.train.batch_size=8\
    train.loaders.train.num_workers=8\
    train.loaders.valid.batch_size=6\
    train.loaders.valid.num_workers=4\
    torch.channels_last=True
done
