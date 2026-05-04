import sys
import yaml
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
try:
    import bitsandbytes as bnb
except ImportError:
    bnb = None
from dataset import DreamBoothDataset


def train(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    subjects = cfg["subjects"]
    if args.subject:
        subjects = [s for s in subjects if s["name"] == args.subject]
        if not subjects:
            raise ValueError(f"Subject '{args.subject}' not found in config.")

    for subject in subjects:
        checkpoint_path = Path(args.output_dir) / subject["name"] / "model_index.json"
        if checkpoint_path.exists():
            print(f"[skip] {subject['name']}: checkpoint already exists", file=sys.stderr)
            continue
        print(f"[train] {subject['name']}", file=sys.stderr)
        _train_one_subject(subject, cfg["hyperparameters"], cfg["model"]["id"], args)


def _train_one_subject(subject, hp, model_id, args):
    device = torch.device(args.device)
    train_dtype = torch.float16 if device.type == "cuda" else torch.float32

    instance_dir = Path(args.instance_dir) / subject["name"]
    class_dir = Path(args.class_dir) / subject["class_name"]
    output_dir = Path(args.output_dir) / subject["name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    instance_prompt = f"a photo of {subject['identifier']} {subject['class_name']}"
    class_prompt = f"a photo of a {subject['class_name']}"

    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
    scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    vae.requires_grad_(False).to(device, dtype=train_dtype)
    vae.eval()
    text_encoder.to(device)  # fp32, trainable
    text_encoder.train()
    unet.to(device)  # fp32 — keeps gradients healthy; only frozen models go fp16
    unet.enable_gradient_checkpointing()

    if bnb is not None and device.type == "cuda":
        optimizer = bnb.optim.AdamW8bit(
            list(unet.parameters()) + list(text_encoder.parameters()),
            lr=hp["learning_rate"],
        )
    else:
        optimizer = torch.optim.AdamW(
            list(unet.parameters()) + list(text_encoder.parameters()),
            lr=hp["learning_rate"],
        )

    dataset = DreamBoothDataset(
        instance_dir=instance_dir,
        instance_prompt=instance_prompt,
        class_dir=class_dir,
        class_prompt=class_prompt,
        tokenizer=tokenizer,
        size=hp["resolution"],
    )
    if len(dataset.instance_paths) == 0:
        raise ValueError(f"No instance images found for subject '{subject['name']}' in {instance_dir}")
    if len(dataset.class_paths) == 0:
        raise ValueError(f"No class images found for class '{subject['class_name']}' in {class_dir}")
    train_batch_size = hp["train_batch_size"]
    if len(dataset) < train_batch_size:
        raise ValueError(
            f"Dataset has only {len(dataset)} samples but train_batch_size is {train_batch_size}. "
            "Add more images or reduce train_batch_size."
        )
    loader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True, drop_last=False)

    unet.train()
    optimizer.zero_grad(set_to_none=True)
    loader_iter = iter(loader)
    step = 0
    while step < hp["max_train_steps"]:
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        instance_pixel_values = batch["instance_pixel_values"].to(device=device, dtype=train_dtype)
        class_pixel_values = batch["class_pixel_values"].to(device=device, dtype=train_dtype)
        instance_input_ids = batch["instance_input_ids"].to(device)
        class_input_ids = batch["class_input_ids"].to(device)

        pixel_values = torch.cat([instance_pixel_values, class_pixel_values], dim=0)
        input_ids = torch.cat([instance_input_ids, class_input_ids], dim=0)
        instance_batch_size = instance_pixel_values.shape[0]

        with torch.no_grad():
            latents = vae.encode(pixel_values).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
        encoder_hidden_states = text_encoder(input_ids)[0]

        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0,
            scheduler.config.num_train_timesteps,
            (latents.shape[0],),
            device=device,
        ).long()
        noisy_latents = scheduler.add_noise(latents, noise, timesteps)

        model_pred = unet(noisy_latents.float(), timesteps, encoder_hidden_states.float()).sample
        if scheduler.config.prediction_type == "v_prediction":
            target = scheduler.get_velocity(latents, noise, timesteps).float()
        else:
            target = noise.float()

        instance_loss = F.mse_loss(model_pred[:instance_batch_size], target[:instance_batch_size])
        prior_loss = F.mse_loss(model_pred[instance_batch_size:], target[instance_batch_size:])
        loss = instance_loss + hp["prior_loss_weight"] * prior_loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        if step % 50 == 0 or step == 1:
            print(f"  step {step:4d}/{hp['max_train_steps']} | loss={loss.item():.4f} (inst={instance_loss.item():.4f}, prior={prior_loss.item():.4f})", file=sys.stderr, flush=True)

    pipeline = StableDiffusionPipeline.from_pretrained(model_id, unet=unet.to("cpu"), text_encoder=text_encoder.to("cpu"), safety_checker=None)
    pipeline.save_pretrained(output_dir)
    print(f"[done] {subject['name']}: saved to {output_dir}", file=sys.stderr, flush=True)
