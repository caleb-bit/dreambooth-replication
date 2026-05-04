import yaml
import json
import torch
from datetime import datetime
from pathlib import Path
from diffusers import StableDiffusionPipeline
from transformers import ViTImageProcessor, ViTModel, CLIPProcessor, CLIPModel

from PIL import Image
import torch.nn.functional as F


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
        from diffusers import UNet2DConditionModel
        from transformers import CLIPTextModel
        unet = UNet2DConditionModel.from_pretrained(
            checkpoint_dir / "unet", torch_dtype=dtype
        )
        text_encoder = CLIPTextModel.from_pretrained(
            checkpoint_dir / "text_encoder", torch_dtype=dtype
        )
        pipe = StableDiffusionPipeline.from_pretrained(
            cfg["model"]["id"],
            unet=unet,
            text_encoder=text_encoder,
            torch_dtype=dtype,
            safety_checker=None,
        ).to(args.device)
        pipe.enable_attention_slicing()

        gen_cfg = cfg["generation"]
        out_root = Path(args.output_dir)
        specific_dir = out_root / "specific" / subject["name"]
        general_dir = out_root / "general" / subject["name"]
        specific_dir.mkdir(parents=True, exist_ok=True)
        general_dir.mkdir(parents=True, exist_ok=True)

        for prompt_idx, template in enumerate(PROMPTS):
            specific_prompt = fill_specific(template, subject["class_name"], subject["identifier"])
            general_prompt = fill_general(template, subject["class_name"])

            for sample_idx in range(gen_cfg["samples_per_prompt"]):
                specific_image = pipe(
                    specific_prompt,
                    num_inference_steps=gen_cfg["num_inference_steps"],
                    guidance_scale=gen_cfg["guidance_scale"],
                ).images[0]
                ts = datetime.now().strftime("%Y%m%dT%H%M%S%f")
                specific_image.save(specific_dir / f"{prompt_idx:02d}_{sample_idx:02d}_{ts}.png")

                general_image = pipe(
                    general_prompt,
                    num_inference_steps=gen_cfg["num_inference_steps"],
                    guidance_scale=gen_cfg["guidance_scale"],
                ).images[0]
                ts = datetime.now().strftime("%Y%m%dT%H%M%S%f")
                general_image.save(general_dir / f"{prompt_idx:02d}_{sample_idx:02d}_{ts}.png")

        print(f"[done] {subject['name']}: eval images saved")
        del pipe


def _compute_metrics(subjects, cfg, args):
    # compute DINO, CLIP-I, CLIP-T metrics as described in Dreambooth paper
    from eval_prompts import PROMPTS, fill_general
    
    # Load Models
    dino_proc = ViTImageProcessor.from_pretrained("facebook/dino-vits16")
    dino_model = ViTModel.from_pretrained("facebook/dino-vits16").to(args.device).eval()
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(args.device).eval()

    results = {}

    for subject in subjects:
        name = subject["name"]
        # Use args.instance_dir (default: dreambooth_dataset) to find reference photos
        ref_path = Path(args.instance_dir) / name
        if not ref_path.exists():
            raise FileNotFoundError(f"Reference image folder not found for subject '{name}': {ref_path}")
        ref_imgs = [Image.open(p).convert("RGB") for p in sorted(ref_path.glob("*")) if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        if not ref_imgs:
            raise ValueError(f"No reference images found for subject '{name}' in {ref_path}")

        # Pre-compute reference features
        with torch.no_grad():
            ref_dino = F.normalize(dino_model(**dino_proc(ref_imgs, return_tensors="pt").to(args.device)).last_hidden_state[:, 0, :], dim=-1)
            ref_clip = F.normalize(clip_model.get_image_features(**clip_proc(ref_imgs, return_tensors="pt").to(args.device)), dim=-1)

        scores = {"dino": [], "clip_i": [], "clip_t": []}
        # Look in args.output_dir (where generate stage saved them)
        gen_dir = Path(args.output_dir) / "specific" / name

        for p_idx, template in enumerate(PROMPTS):
            # Prompt for CLIP-T
            txt_inputs = clip_proc(text=[fill_general(template, subject["class_name"])], return_tensors="pt", padding=True).to(args.device)
            with torch.no_grad():
                txt_feat = F.normalize(clip_model.get_text_features(**txt_inputs), dim=-1)

            for s_idx in range(cfg["generation"]["samples_per_prompt"]):
                exact_img_path = gen_dir / f"{p_idx:02d}_{s_idx:02d}.png"
                if exact_img_path.exists():
                    img_path = exact_img_path
                else:
                    candidates = sorted(
                        gen_dir.glob(f"{p_idx:02d}_{s_idx:02d}_*.png"),
                        key=lambda p: p.stat().st_mtime,
                        reverse=True,
                    )
                    if not candidates:
                        continue
                    img_path = candidates[0]

                gen_img = Image.open(img_path).convert("RGB")
                with torch.no_grad():
                    g_dino = F.normalize(dino_model(**dino_proc(gen_img, return_tensors="pt").to(args.device)).last_hidden_state[:, 0, :], dim=-1)
                    g_clip = F.normalize(clip_model.get_image_features(**clip_proc(gen_img, return_tensors="pt").to(args.device)), dim=-1)

                scores["dino"].append(torch.mm(g_dino, ref_dino.t()).mean().item())
                scores["clip_i"].append(torch.mm(g_clip, ref_clip.t()).mean().item())
                scores["clip_t"].append(torch.mm(g_clip, txt_feat.t()).item())

        results[name] = {k: sum(v)/len(v) for k, v in scores.items() if v}

    # Write results to args.results_dir
    res_out = Path(args.results_dir) / "metrics.json"
    res_out.parent.mkdir(parents=True, exist_ok=True)
    with open(res_out, "w") as f:
        json.dump(results, f, indent=4)