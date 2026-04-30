import yaml
import json
import torch
from pathlib import Path
from diffusers import StableDiffusionPipeline
from transformers import AutoProcessor, AutoModel
from utils import save_image


def eval(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    subjects = cfg["subjects"]
    if args.subject:
        subjects = [s for s in subjects if s["name"] == args.subject]
        if not subjects:
            raise ValueError(f"Subject '{args.subject}' not found in config.")

    if args.stage == "generate":
        _generate_eval_images(subjects, cfg, args)
    elif args.stage == "metrics":
        _compute_metrics(subjects, cfg, args)


def _generate_eval_images(subjects, cfg, args):
    from eval_prompts import PROMPTS, fill_specific, fill_general

    for subject in subjects:
        checkpoint_dir = Path(args.checkpoint_dir) / subject["name"]
        if not checkpoint_dir.exists():
            print(f"[skip] {subject['name']}: no checkpoint found at {checkpoint_dir}")
            continue

        print(f"[eval:generate] {subject['name']}")

        dtype = torch.float16 if args.device == "cuda" else torch.float32
        pipe = StableDiffusionPipeline.from_pretrained(
            checkpoint_dir,
            torch_dtype=dtype,
            safety_checker=None,
        ).to(args.device)
        pipe.enable_attention_slicing()

        gen_cfg = cfg["generation"]

        for prompt_idx, template in enumerate(PROMPTS):
            specific_prompt = fill_specific(template, subject["class_name"], subject["identifier"])
            general_prompt = fill_general(template, subject["class_name"])

            for sample_idx in range(gen_cfg["samples_per_prompt"]):
                specific_image = pipe(
                    specific_prompt,
                    num_inference_steps=gen_cfg["num_inference_steps"],
                    guidance_scale=gen_cfg["guidance_scale"],
                ).images[0]
                save_image(
                    f"eval/specific/{subject['name']}",
                    f"{prompt_idx:02d}_{sample_idx:02d}",
                    specific_image,
                )

                general_image = pipe(
                    general_prompt,
                    num_inference_steps=gen_cfg["num_inference_steps"],
                    guidance_scale=gen_cfg["guidance_scale"],
                ).images[0]
                save_image(
                    f"eval/general/{subject['name']}",
                    f"{prompt_idx:02d}_{sample_idx:02d}",
                    general_image,
                )

        print(f"[done] {subject['name']}: eval images saved")
        del pipe


def _compute_metrics(subjects, cfg, args):
    pass
