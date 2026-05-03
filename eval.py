import yaml
import json
import torch
from pathlib import Path
from diffusers import StableDiffusionPipeline
from transformers import AutoProcessor, AutoModel, ViTImageProcessor, ViTModel, CLIPProcessor, CLIPModel
from utils import save_image

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
        ref_imgs = [Image.open(p).convert("RGB") for p in ref_path.glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        
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
                img_path = gen_dir / f"{p_idx:02d}_{s_idx:02d}.png"
                if not img_path.exists(): continue
                
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