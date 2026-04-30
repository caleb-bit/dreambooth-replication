import yaml
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
import bitsandbytes as bnb
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
            print(f"[skip] {subject['name']}: checkpoint already exists")
            continue
        print(f"[train] {subject['name']}")
        _train_one_subject(subject, cfg["hyperparameters"], cfg["model"]["id"], args)


def _train_one_subject(subject, hp, model_id, args):
    device = args.device

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

    vae.requires_grad_(False).to(device, dtype=torch.float16)
    text_encoder.requires_grad_(False).to(device, dtype=torch.float16)
    unet.to(device)
    unet.enable_gradient_checkpointing()

    optimizer = bnb.optim.AdamW8bit(unet.parameters(), lr=hp["learning_rate"])

    dataset = DreamBoothDataset(
        instance_dir=instance_dir,
        instance_prompt=instance_prompt,
        class_dir=class_dir,
        class_prompt=class_prompt,
        tokenizer=tokenizer,
        size=hp["resolution"],
    )
    loader = DataLoader(dataset, batch_size=hp["train_batch_size"], shuffle=True, drop_last=True)

    step = 0
    while step < hp["max_train_steps"]:
        pass  # training loop goes here

    pipeline = StableDiffusionPipeline.from_pretrained(model_id, unet=unet, safety_checker=None)
    pipeline.save_pretrained(output_dir)
    print(f"[done] {subject['name']}: saved to {output_dir}")
