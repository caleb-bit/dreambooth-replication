import yaml
import torch
from diffusers import StableDiffusionPipeline
from utils import save_image


def generate(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    pipe = StableDiffusionPipeline.from_pretrained(
        cfg["model_id"],
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to(cfg["device"])
    pipe.enable_attention_slicing()

    generator = torch.Generator(device=cfg["device"]).manual_seed(cfg["seed"])

    for class_name in cfg["classes"]:
        prompt = cfg["prompt_template"].format(class_name=class_name)
        print(f"[generate] {class_name}: generating {cfg['num_images']} images")

        for i in range(cfg["num_images"]):
            image = pipe(
                prompt,
                num_inference_steps=cfg["num_inference_steps"],
                guidance_scale=cfg["guidance_scale"],
                generator=generator,
            ).images[0]
            path = save_image(f"class_images/{class_name}", f"{i:04d}", image)

        print(f"[done] {class_name}: saved to {path.parent}")
