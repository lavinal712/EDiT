# EDiT

EDiT is a diffusion transformer (DiT) model architecture. For now, *E* means *editted* as it supports text prompts instead of class labels.

EDiT adopts SDXL-VAE and CLIP to encode images and text.

- [SDXL-VAE](https://huggingface.co/stabilityai/sdxl-vae)
- [CLIP-ViT-L/14](https://huggingface.co/openai/clip-vit-large-patch14)

This repository contains:

- 🪐 A simple PyTorch implementation of EDiT
- 🛸 Training script on ImageNet with text prompt.

## Setup

Please refer to DiT and PixArt-α.

## Training

### Training EDiT

```bash
accelerate launch --mixed_precision fp16 train.py --data_path /path/to/ImageNet/train
```

## TODO

- [x] Support text prompt for DiT
- [x] Training script using [accelerate](https://github.com/huggingface/accelerate)
- [ ] [Gradio](https://www.gradio.app/) for inference

## Acknowledgments

EDiT has been greatly inspired by the following amazing works and teams:

- [DiT](https://github.com/facebookresearch/DiT): Scalable Diffusion Models with Transformers
- [fast-DiT](https://github.com/chuanyangjin/fast-DiT): Scalable Diffusion Models with Transformers
- [PixArt-α](https://github.com/PixArt-alpha/PixArt-alpha/): PixArt-α: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis
