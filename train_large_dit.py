import argparse
import contextlib
import json
import os
import socket
from copy import deepcopy
from datetime import datetime
from glob import glob
from time import time

import torch
import torch.distributed as dist
from accelerate import Accelerator
from diffusers.models import AutoencoderKL
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data.imagenet import CustomImageNetDataset, ImageNetDataset
from diffusion import create_diffusion
from models.ema import update_ema
from models.modeling_large_dit import DiT_Llama_models
from models.utils import requires_grad
from utils.logger import create_logger

# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def get_train_sampler(dataset, rank, world_size, global_batch_size, max_steps,
                      resume_step, seed):
    sample_indices = torch.empty([max_steps * global_batch_size // world_size],
                                 dtype=torch.long)
    epoch_id, fill_ptr, offs = 0, 0, 0
    while fill_ptr < sample_indices.size(0):
        g = torch.Generator()
        g.manual_seed(seed + epoch_id)
        epoch_sample_indices = torch.randperm(len(dataset), generator=g)
        epoch_id += 1
        epoch_sample_indices = epoch_sample_indices[
            (rank + offs) % world_size::world_size
        ]
        offs = (offs + world_size - len(dataset) % world_size) % world_size
        epoch_sample_indices = epoch_sample_indices[
            :sample_indices.size(0) - fill_ptr
        ]
        sample_indices[fill_ptr: fill_ptr + epoch_sample_indices.size(0)] = \
            epoch_sample_indices
        fill_ptr += epoch_sample_indices.size(0)
    return sample_indices[resume_step * global_batch_size // world_size:].tolist()


def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), (
        "Training currently requires at least one GPU."
    )

    # Setup accelerator:
    accelerator = Accelerator()
    device = accelerator.device
    world_size = accelerator.num_processes
    rank = accelerator.process_index

    assert args.global_batch_size % world_size == 0, (
        "Batch size must be divisible by data parrallel world size."
    )
    local_batch_size = args.global_batch_size // world_size
    seed = args.global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    # Setup an experiment folder:
    if accelerator.is_main_process:
        os.makedirs(args.results_dir, exist_ok=True)
        checkpoint_dir = os.path.join(args.results_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(args.results_dir)
        logger.info(f"Experiment directory: {args.results_dir}")
        tb_logger = SummaryWriter(os.path.join(
            args.results_dir, "tensorboard",
            datetime.now().strftime("%Y%m%d_%H%M%S_") + socket.gethostname()
        ))

    if accelerator.is_main_process:
        logger.info("Training arguments: " + json.dumps(args.__dict__, indent=2))

    # Create model:
    assert args.image_size % 8 == 0, (
        "Image size must be divisible by 8 (for the VAE encoder)."
    )
    latent_size = args.image_size // 8
    model = DiT_Llama_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        qk_norm=args.qk_norm,
    ).to(device)
    if accelerator.is_main_process:
        logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    if args.auto_resume and args.resume is None:
        try:
            existing_checkpoints = os.listdir(checkpoint_dir)
            if len(existing_checkpoints) > 0:
                existing_checkpoints.sort()
                args.resume = os.path.join(checkpoint_dir,
                                           existing_checkpoints[-1])
        except Exception:
            pass
        if args.resume is not None:
            if accelerator.is_main_process:
                logger.info(f"Auto resuming from: {args.resume}")

    # Note that parameter initialization is done within the DiT constructor
    model_ema = deepcopy(model).to(device)
    if args.resume:
        if accelerator.is_main_process:  # other ranks receive weights in setup_fsdp_sync
            logger.info(f"Resuming model weights from: {args.resume}")
            model.load_state_dict(torch.load(os.path.join(
                args.resume,
                f"consolidated.{rank:02d}-of-{world_size:02d}.pth",
            ), map_location="cpu"), strict=True)
            logger.info(f"Resuming ema weights from: {args.resume}")
            model_ema.load_state_dict(torch.load(os.path.join(
                args.resume,
                f"consolidated_ema.{rank:02d}-of-{world_size:02d}.pth",
            ), map_location="cpu"), strict=True)
    elif args.init_from:
        if accelerator.is_main_process:
            logger.info(f"Initializing model weights from: {args.init_from}")
            state_dict = torch.load(os.path.join(
                args.init_from,
                f"consolidated.{rank:02d}-of-{world_size:02d}.pth",
            ), map_location="cpu")
            missing_keys, unexpected_keys = \
                model.load_state_dict(state_dict, strict=False)
            missing_keys_ema, unexpected_keys_ema = \
                model_ema.load_state_dict(state_dict, strict=False)
            del state_dict
            assert set(missing_keys) == set(missing_keys_ema)
            assert set(unexpected_keys) == set(unexpected_keys_ema)
            logger.info("Model initialization result:")
            logger.info(f"  Missing keys: {missing_keys}")
            logger.info(f"  Unexpected keys: {unexpected_keys}")

    # default: 1000 steps, linear noise schedule
    diffusion = create_diffusion(timestep_respacing="")
    vae = AutoencoderKL.from_pretrained(
        f"stabilityai/sd-vae-ft-{args.vae}"
        if args.local_diffusers_model_root is None else
        os.path.join(args.local_diffusers_model_root,
                     f"stabilityai/sd-vae-ft-{args.vae}")
    ).to(device)

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant
    # learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(),
                            lr=args.lr, weight_decay=args.wd)
    if args.resume:
        opt_state_world_size = len([
            x for x in os.listdir(args.resume)
            if x.startswith("optimizer.") and x.endswith(".pth")
        ])
        assert opt_state_world_size == world_size, (
            f"Resuming from a checkpoint with unmatched world size "
            f"({world_size} vs. {opt_state_world_size}) "
            f"is currently not supported."
        )
        logger.info(f"Resuming optimizer states from: {args.resume}")
        opt.load_state_dict(torch.load(os.path.join(
            args.resume,
            f"optimizer.{rank:05d}-of-"
            f"{world_size:05d}.pth",
        ), map_location="cpu"))
        for param_group in opt.param_groups:
            param_group["lr"] = args.lr
            param_group["weight_decay"] = args.wd

        with open(os.path.join(args.resume, "resume_step.txt")) as f:
            resume_step = int(f.read().strip())
    else:
        resume_step = 0

    # Setup data:
    dataset = ImageNetDataset(args.data_path, transform="center_crop")
    num_samples = args.global_batch_size * args.max_steps
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")
        logger.info(f"Total # samples to consume: {num_samples:,} "
                    f"({num_samples / len(dataset):.2f} epochs)")
    loader = DataLoader(
        dataset,
        batch_size=local_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Prepare models for training:
    # important! This enables embedding dropout for classifier-free guidance
    update_ema(model_ema, model, decay=0)
    model.train()
    model_ema.eval()
    model, opt, loader = accelerator.prepare(model, opt, loader)

    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    start_time = time()

    if accelerator.is_main_process:
        logger.info(f"Training for {args.max_steps:,} steps...")
    for step, (x, y) in enumerate(loader, start=resume_step):
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            # Map input images to latent space + normalize latents:
            x = vae.encode(x).latent_dist.sample().mul_(0.18215)
        t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],),
                          device=device)

        loss_item = 0.
        opt.zero_grad()
        for mb_idx in range(
            (local_batch_size - 1) // args.micro_batch_size + 1
        ):
            mb_st = mb_idx * args.micro_batch_size
            mb_ed = min((mb_idx + 1) * args.micro_batch_size,
                        local_batch_size)
            last_mb = (mb_ed == local_batch_size)

            x_mb = x[mb_st: mb_ed]
            y_mb = y[mb_st: mb_ed]
            t_mb = t[mb_st: mb_ed]

            model_kwargs = dict(y=y_mb)
            with {
                "bf16": torch.cuda.amp.autocast(dtype=torch.bfloat16),
                "fp16": torch.cuda.amp.autocast(dtype=torch.float16),
                "fp32": contextlib.nullcontext(),
                "tf32": contextlib.nullcontext(),
            }[args.precision]:
                loss_dict = diffusion.training_losses(
                    model, x_mb, t_mb, model_kwargs
                )
            loss = loss_dict["loss"].sum() / local_batch_size
            loss_item += loss.item()
            with (
                model.no_sync()
                if args.data_parallel in ["sdp", "hsdp"] and not last_mb else
                contextlib.nullcontext()
            ):
                accelerator.backward(loss)

        accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)

        if accelerator.is_main_process and tb_logger is not None:
            tb_logger.add_scalar("train/loss", loss_item, step)
            tb_logger.add_scalar("train/lr", opt.param_groups[0]["lr"], step)

        opt.step()
        update_ema(model_ema, model)

        # Log loss values:
        running_loss += loss_item
        log_steps += 1
        if (step + 1) % args.log_every == 0:
            # Measure training speed:
            torch.cuda.synchronize()
            end_time = time()
            secs_per_step = (end_time - start_time) / log_steps
            imgs_per_sec = args.global_batch_size * log_steps / (
                end_time - start_time
            )
            # Reduce loss history over all processes:
            avg_loss = torch.tensor(running_loss / log_steps,
                                    device=device)
            avg_loss = avg_loss.item() / accelerator.num_processes
            if accelerator.is_main_process:
                logger.info(f"(step={step + 1:07d}) "
                            f"Train Loss: {avg_loss:.4f}, "
                            f"Train Secs/Step: {secs_per_step:.2f}, "
                            f"Train Imgs/Sec: {imgs_per_sec:.2f}")
            # Reset monitoring variables:
            running_loss = 0
            log_steps = 0
            start_time = time()

        # Save DiT checkpoint:
        if (
            (step + 1) % args.ckpt_every == 0
            or (step + 1) == args.max_steps
        ):
            if accelerator.is_main_process: 
                checkpoint_path = f"{checkpoint_dir}/{step + 1:07d}"
                os.makedirs(checkpoint_path, exist_ok=True)

                consolidated_model_state_dict = model.state_dict()
                consolidated_fn = (
                    "consolidated."
                    f"{rank:02d}-of-"
                    f"{world_size:02d}"
                    ".pth"
                )
                torch.save(
                    consolidated_model_state_dict,
                    os.path.join(checkpoint_path, consolidated_fn),
                )
                del consolidated_model_state_dict
                logger.info(f"Saved consolidated to {checkpoint_path}.")

                consolidated_ema_state_dict = model_ema.state_dict()
                consolidated_ema_fn = (
                    "consolidated_ema."
                    f"{rank:02d}-of-"
                    f"{world_size:02d}"
                    ".pth"
                )
                torch.save(
                    consolidated_ema_state_dict,
                    os.path.join(checkpoint_path, consolidated_ema_fn),
                )
                del consolidated_ema_state_dict
                logger.info(f"Saved consolidated_ema to {checkpoint_path}.")

                opt_state_fn = (
                    f"optimizer.{rank:05d}-of-"
                    f"{world_size:05d}.pth"
                )
                torch.save(opt.state_dict(),
                           os.path.join(checkpoint_path, opt_state_fn))
                logger.info(f"Saved optimizer to {checkpoint_path}.")

                torch.save(args,
                           os.path.join(checkpoint_path, "model_args.pth"))
                with open(
                    os.path.join(checkpoint_path, "resume_step.txt"), "w"
                ) as f:
                    print(step + 1, file=f)
                logger.info(f"Saved training arguments to {checkpoint_path}.")

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    if accelerator.is_main_process:
        logger.info("Done!")


if __name__ == "__main__":
    # Default args here will train DiT_Llama2_7B_patch2 with the
    # hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--model", type=str, choices=list(DiT_Llama_models.keys()),
                        default="DiT_Llama2_7B_patch2")
    parser.add_argument("--image_size", type=int, choices=[256, 512],
                        default=256)
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument(
        "--max_steps", type=int, default=100_000,
        help="Number of training steps."
    )
    parser.add_argument("--global_batch_size", type=int, default=256)
    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument("--global_seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"],
                        default="ema")  # Choice doesn't affect training
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--ckpt_every", type=int, default=50_000)
    parser.add_argument("--master_port", type=int, default=8964)
    parser.add_argument("--data_parallel", type=str,
                        choices=["sdp", "fsdp"], default="fsdp")
    parser.add_argument("--precision",
                        choices=["fp32", "tf32", "fp16", "bf16"],
                        default="bf16")
    parser.add_argument("--grad_precision",
                        choices=["fp32", "fp16", "bf16"])
    parser.add_argument(
        "--local_diffusers_model_root", type=str,
        help="Specify the root directory if diffusers models are to be loaded "
             "from the local filesystem (instead of being automatically "
             "downloaded from the Internet). Useful in environments without "
             "Internet access."
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="Learning rate."
    )
    parser.add_argument(
        "--no_auto_resume", action="store_false", dest="auto_resume",
        help="Do NOT auto resume from the last checkpoint in --results_dir."
    )
    parser.add_argument(
        "--resume", type=str,
        help="Resume training from a checkpoint folder."
    )
    parser.add_argument(
        "--init_from", type=str,
        help="Initialize the model weights from a checkpoint folder. "
             "Compared to --resume, this loads neither the optimizer states "
             "nor the data loader states."
    )
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=2.0,
        help="Clip the L2 norm of the gradients to the given value."
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=0.,
        help="Weight decay for the optimizer.",
    )
    parser.add_argument(
        "--qk_norm",
        action="store_true",
    )
    args = parser.parse_args()

    main(args)
