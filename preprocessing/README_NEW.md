<h1 align="center"> Preprocessing Guide
</h1>

#### Dataset download

```bash
# Convert raw ImageNet data to a ZIP archive at 256x256 resolution
torchrun --nproc_per_node=8 preprocess.py --convert --source=[YOUR_DOWNLOAD_PATH]/ILSVRC/Data/CLS-LOC/train \
    --dest=[TARGET_PATH]/images --resolution=256 --transform=center-crop-dhariwal
```

```bash
# Speed up the preprocessing process 8x
torchrun --nproc_per_node=8 preprocess.py --source=[YOUR_DOWNLOAD_PATH]/ILSVRC/Data/CLS-LOC/train \
    --dest=[TARGET_PATH]/vae-sd \
    --dest-images=[TARGET_PATH]/images \
    --batch-size=128 \
    --resolution=256 \
    --transform=center-crop-dhariwal
```

Here,`YOUR_DOWNLOAD_PATH` is the directory that you downloaded the dataset, and `TARGET_PATH` is the directory that you will save the preprocessed images and corresponding compressed latent vectors. This directory will be used for your experiment scripts. 
