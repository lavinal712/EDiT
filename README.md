# EDiT

## About The Project

This repository provides code for training **<font color="#7799CC">Diffusion Transformer</font>** (DiT) models.

- Improved DiT training script with `accelerate`
- Supports multiple models.
- Easy to modify the code for your own research.

## Supported Models

| Model | Training | Inference |
|-------|----------|-----------|
| [DiT](https://github.com/facebookresearch/DiT) | ✅ | ✅ |
| [SiT](https://github.com/willisma/SiT) | ✅ | ✅ |
| [REPA](https://github.com/sihyun-yu/REPA) | ✅ | ✅ |
| [Large-DiT](https://github.com/Alpha-VLLM/LLaMA2-Accessory/tree/main/Large-DiT-ImageNet) | ❌ | ✅ |

## Getting Started

### Installation

```bash
conda create -n edit python=3.11 -y
conda activate edit
pip install -r requirements.txt
```

## Acknowledgments

- [DiT](https://github.com/facebookresearch/DiT): Scalable Diffusion Models with Transformers
- [fast-DiT](https://github.com/chuanyangjin/fast-DiT): Scalable Diffusion Models with Transformers
- [SiT](https://github.com/willisma/SiT): SiT: Exploring Flow and Diffusion-based Generative Models with Scalable Interpolant Transformers
- [REPA](https://github.com/sihyun-yu/REPA): Representation Alignment for Generation:
Training Diffusion Transformers Is Easier Than You Think
- [Large-DiT](https://github.com/Alpha-VLLM/LLaMA2-Accessory/tree/main/Large-DiT-ImageNet): Large Diffusion Transformer
- [PixArt-α](https://github.com/PixArt-alpha/PixArt-alpha/): PixArt-α: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis
- [LlamaGen](https://github.com/FoundationVision/LlamaGen): Autoregressive Model Beats Diffusion: Llama for Scalable Image Generation
