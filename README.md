# Replicating DreamBooth: Subject-Driven Fine-Tuning of Stable Diffusion

**Ryan Qiu, Gordon Mei, Caleb Shim, Evan Cui** · Cornell University · CS4782 Deep Learning

A from-scratch replication of [DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://arxiv.org/abs/2208.12242) (Ruiz et al., CVPR 2023) on Stable Diffusion v1.5. Given 3–5 reference photos, DreamBooth fine-tunes a diffusion model to bind a specific subject to a rare identifier token, enabling generation of that subject in novel contexts via text prompts.

---

## Chosen Result

We reproduce **Table 2** of the DreamBooth paper: subject fidelity (DINO, CLIP-I) and prompt fidelity (CLIP-T) for fine-tuned SD 1.5 models. The paper reports DINO 0.668 / CLIP-I 0.803 / CLIP-T 0.305 for SD, demonstrating that subject identity and prompt-following can coexist in a fine-tuned diffusion model.

---

## GitHub Contents

```
code/        Training, evaluation, and generation scripts + configs
data/        Dataset info and download instructions
artifacts/   Subject images (Git LFS) and generated class images
results/     Metric JSONs and qualitative sample outputs
poster/      Conference-style poster (PDF)
report/      2-page project summary report (PDF)
```

---

## Re-implementation Details

- **Model:** Stable Diffusion v1.5 (`runwayml/stable-diffusion-v1-5`). UNet (~860M params) fine-tuned; VAE and CLIP text encoder frozen.
- **Training objective:** Combined instance loss + prior-preservation loss weighted by λ=1.0 to prevent language drift.
- **Rare-token identifier:** `sks` — a token with negligible prior semantics used to bind the subject.
- **Dataset:** `dog`, `dog2`, `backpack` from [google/dreambooth](https://github.com/google/dreambooth); 5 instance images and 100 SD-generated class images per subject.
- **Metrics:** DINO (ViT-S/16), CLIP-I and CLIP-T (CLIP ViT-L/14).
- **Extensions:** Logo fine-tuning (CS4782 course logo); style transfer via Van Gogh paintings (600 steps, lr 2×10⁻⁶).
- **Key finding:** UNet must be kept in FP32 — FP16 causes silent gradient underflow with no visible error.

---

## Reproduction Steps

**Requirements:** Python 3.10+, CUDA GPU (T4 or better recommended), ~35 min per subject.

```bash
git clone https://github.com/caleb-bit/dreambooth-replication.git
cd dreambooth-replication
git lfs pull
pip install -r code/requirements.txt
```

**Step 1 — Generate class regularization images** (skip if using the provided `artifacts/class_images/`):
```bash
python code/main.py generate --config code/configs/generate.yaml
```

**Step 2 — Fine-tune on a subject:**
```bash
python code/main.py train \
    --config code/configs/subjects.yaml \
    --subject dog \
    --instance-dir artifacts/subject_images \
    --class-dir artifacts/class_images \
    --output-dir checkpoints \
    --device cuda
```

**Step 3 — Generate evaluation images from the checkpoint:**
```bash
python code/main.py eval \
    --config code/configs/subjects.yaml \
    --stage generate \
    --subject dog \
    --checkpoint-dir checkpoints \
    --instance-dir artifacts/subject_images \
    --output-dir artifacts/eval \
    --device cuda
```

**Step 4 — Compute DINO / CLIP-I / CLIP-T metrics:**
```bash
python code/main.py eval \
    --config code/configs/subjects.yaml \
    --stage metrics \
    --subject dog \
    --checkpoint-dir checkpoints \
    --instance-dir artifacts/subject_images \
    --output-dir artifacts/eval \
    --results-dir results \
    --device cpu
```

---

## Results / Insights

| Subject | DINO ↑ | CLIP-I ↑ | CLIP-T ↑ |
|---|---|---|---|
| dog | 0.716 | 0.867 | 0.275 |
| dog2 | 0.542 | 0.828 | 0.247 |
| backpack | 0.536 | 0.887 | 0.274 |
| **Ours (mean)** | **0.598** | **0.861** | **0.265** |
| Paper (SD 1.5) | 0.668 | 0.803 | 0.305 |

Our CLIP-I (0.861) exceeds the paper's reported value; DINO and CLIP-T fall slightly below, consistent with evaluating 3 subjects rather than the paper's full 30.

![Reference dog and generated output](results/samples/sksdogatcornell.png)

---

## Conclusion

Full UNet fine-tuning with prior preservation achieves strong subject fidelity in under 40 minutes on a free GPU. Three implementation details proved critical: (1) prior preservation is necessary — ablating it causes complete identity collapse; (2) FP16 training of the UNet causes silent gradient underflow with no error or warning; (3) freezing the text encoder prevents overfitting and improves compositional outputs. Logo fine-tuning reproduced the graphic in direct generation but failed at context placement, with the model rendering the identifier as literal glyphs. Style transfer via Van Gogh fine-tuning qualitatively succeeded with reduced hyperparameters, with the model retaining generic painting ability when the identifier was omitted.

---

## References

- Ruiz et al. *DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation.* CVPR 2023.
- Rombach et al. *High-Resolution Image Synthesis with Latent Diffusion Models.* CVPR 2022.
- Radford et al. *Learning Transferable Visual Models From Natural Language Supervision.* ICML 2021.
- Caron et al. *Emerging Properties in Self-Supervised Vision Transformers.* ICCV 2021.
- HuggingFace Diffusers: https://github.com/huggingface/diffusers
- DreamBooth Dataset: https://github.com/google/dreambooth
- MoMA — Vincent van Gogh: https://www.moma.org/artists/2206-vincent-van-gogh
- National Gallery of Art — Vincent van Gogh: https://www.nga.gov/collection/artist-info.1349.html
- Art Institute of Chicago — The Bedroom: https://www.artic.edu/artworks/28560/the-bedroom

---

## Acknowledgements

This project was completed as part of **CS4782: Deep Learning** at Cornell University (Spring 2025). Training was performed on Google Colab (T4 GPU). We used the HuggingFace `diffusers` library and the official `google/dreambooth` dataset.
