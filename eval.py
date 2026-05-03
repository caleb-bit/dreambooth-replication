import yaml
import json
import re
import torch
from pathlib import Path
from diffusers import StableDiffusionPipeline
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
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
    from eval_prompts import PROMPTS, fill_specific

    device = args.device

    clip_name = "openai/clip-vit-base-patch32"
    try:
        processor = CLIPProcessor.from_pretrained(clip_name)
        clip_model = CLIPModel.from_pretrained(clip_name).to(device)
        clip_model.eval()
    except Exception as e:
        print(f"[metrics] failed to load CLIP model '{clip_name}': {e}")
        return

    results = {}

    def embed_images(paths, batch_size=16):
        embs = []
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i : i + batch_size]
            images = []
            for p in batch_paths:
                with Image.open(p) as image:
                    images.append(image.convert("RGB"))
            inputs = processor(images=images, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                emb = clip_model.get_image_features(**inputs)
                emb = emb / emb.norm(p=2, dim=1, keepdim=True)
            embs.append(emb)
        return torch.cat(embs, dim=0)

    def _latest_images(paths):
        """Return only the most recent image per prompt/sample prefix (e.g. '00_01')."""
        by_prefix = {}
        for p in paths:
            m = re.match(r"(\d{2}_\d{2})_", p.name)
            prefix = m.group(1) if m else p.stem
            if prefix not in by_prefix or p.name > by_prefix[prefix].name:
                by_prefix[prefix] = p
        return sorted(by_prefix.values())

    for subject in subjects:
        name = subject["name"]
        print(f"[metrics] {name}")

        configured_gen_dir = Path(args.output_dir) / "specific" / name
        artifacts_gen_dir = Path("artifacts") / "eval" / "specific" / name
        gen_dir = configured_gen_dir if configured_gen_dir.exists() else artifacts_gen_dir
        gen_paths = _latest_images(sorted(gen_dir.glob("*.png")))
        if not gen_paths:
            print(f"[skip] {name}: no generated images at {gen_dir}")
            continue

        ref_dir = Path(args.instance_dir) / name
        ref_paths = sorted(ref_dir.glob("*.[jp][pn]g"))
        if not ref_paths:
            print(f"[skip] {name}: no reference images at {ref_dir}")
            continue

        # embeddings
        ref_emb = embed_images(ref_paths)
        gen_emb = embed_images(gen_paths)

        # CLIP-I: for each generated image, take max similarity to any reference image
        sims = (gen_emb @ ref_emb.T).numpy()
        max_per_gen = sims.max(axis=1)
        clip_i = float(max_per_gen.mean())

        # CLIP-T: compute similarity between each generated image and its prompt
        prompts = []
        for p in gen_paths:
            m = re.search(r"(\d{2})_(\d{2})", p.name)
            prompt_idx = int(m.group(1)) if m else 0
            prompts.append(fill_specific(PROMPTS[prompt_idx], subject["class_name"], subject.get("identifier", "")))

        text_inputs = processor(text=prompts, return_tensors="pt", padding=True)
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        with torch.no_grad():
            txt_emb = clip_model.get_text_features(**text_inputs)
            txt_emb = txt_emb / txt_emb.norm(p=2, dim=1, keepdim=True)

        gen_emb_dev = gen_emb.to(device)
        per_image_sim = (gen_emb_dev * txt_emb).sum(dim=1).cpu().numpy()
        clip_t = float(per_image_sim.mean())

        results[name] = {"clip_i": clip_i, "clip_t": clip_t}

        print(f"[metrics] {name}: CLIP-I={clip_i:.4f} CLIP-T={clip_t:.4f}")

    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "metrics.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[metrics] written to {out_path}")
