import yaml
import torch
from diffusers import StableDiffusionPipeline
from utils import save_image


def generate(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # MPS doesn't support fp16 — use fp32 there; fp16 on CUDA only
    dtype = torch.float16 if cfg["device"] == "cuda" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        cfg["model_id"],
        torch_dtype=dtype,
        safety_checker=None,
    ).to(cfg["device"])
    pipe.enable_attention_slicing()

    generator = torch.Generator(device=cfg["device"]).manual_seed(cfg["seed"])

    for class_name in cfg["classes"]:
        prompt = cfg["prompt_template"].format(class_name=class_name)
        print(f"[generate] {class_name}: generating {cfg['num_images']} images")

        batch_size = cfg.get("generation_batch_size", 4)
        i = 0
        while i < cfg["num_images"]:
            n = min(batch_size, cfg["num_images"] - i)
            images = pipe(
                prompt,
                num_images_per_prompt=n,
                num_inference_steps=cfg["num_inference_steps"],
                guidance_scale=cfg["guidance_scale"],
                generator=generator,
            ).images
            for img in images:
                path = save_image(f"class_images/{class_name}", f"{i:04d}", img)
                i += 1

        print(f"[done] {class_name}: saved to {path.parent}")
