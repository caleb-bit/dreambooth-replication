# dreambooth-replication

## Running on Colab

### Setup (run once per session)

```python
# Cell 1 — clone & install
!git clone https://github.com/caleb-bit/dreambooth-replication.git
%cd dreambooth-replication
!git lfs pull   # pull subject/class images stored in LFS
!pip install -q diffusers transformers accelerate bitsandbytes Pillow torchvision
```

```python
# Cell 2 — mount Drive (keeps checkpoints across sessions)
from google.colab import drive
drive.mount("/content/drive")

WORK_DIR = "/content/drive/MyDrive/dreambooth"   # change to your Drive path
!mkdir -p "{WORK_DIR}/checkpoints"
```

### Training

```python
# Cell 3 — train a subject
!python main.py train \
    --config configs/subjects.yaml \
    --subject dog \
    --instance-dir artifacts/subject_images \
    --class-dir artifacts/class_images \
    --output-dir "{WORK_DIR}/checkpoints" \
    --device cuda
```

Logs print every 50 steps to stderr. If the session disconnects and you rerun, already-completed subjects are skipped automatically.

### Generating a test image from a checkpoint

```python
import torch
from diffusers import StableDiffusionPipeline

CHECKPOINT = "/content/drive/MyDrive/dreambooth/checkpoints/dog"
PROMPT = "a photo of sks dog on the beach"

pipe = StableDiffusionPipeline.from_pretrained(
    CHECKPOINT, torch_dtype=torch.float16, safety_checker=None
).to("cuda")
pipe.enable_attention_slicing()

image = pipe(
    PROMPT,
    num_inference_steps=50,
    guidance_scale=7.5,
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]

image.save("test_output.png")
image  # displays inline
```

---

Ryan Qiu, Gordon Mei, Caleb Shim, Evan Cui

> Note: Subject = specific dog. Class = dogs

`main.py` contains commands to run different parts of the project. I separated this into three parts

**paths are relative to where the command is run**

```
python main.py generate --config=CONFIGPATH
```

training uses ~200 images of the broader class (e.g. "dog") as a regularization set. This command generates those using the base SD model. settings are in `configs/generate.yaml`

```
python main.py train \
 --config=CONFIGPATH  \
 --subject=SUBJECT \
 --instance-dir=INSTANCEDIR \
 --class-dir=CLASSDIR \
 --output-dir=OUTPUTDIR \
 --device=DEVICE \
```

This spins up the training loop defined in `train.py` with the given config, subject class, etc.

subject should be something like "cat6" or "dog2". They must match a name in the config.

The script looks for instance images (jpg or png) at `INSTANCEDIR/SUBJECT/` and class images at `CLASSDIR/CLASSNAME/`.

Skips subjects whose checkpoint already exists in `OUTPUTDIR/`, so it's safe to rerun after a crash. You should use the path to subjects.yaml for CONFIG PATH.

```
# Stage 1 — generate images from checkpoints
!python main.py eval \
 --config=/content/dreambooth-replication/configs/subjects.yaml \
 --stage=generate \
 --subject=dog \
 --checkpoint-dir="/content/drive/MyDrive/dreambooth/checkpoints" \
 --instance-dir=artifacts/subject_images \
 --output-dir="/content/drive/MyDrive/dreambooth/eval_output" \
 --device=cuda
```

```
# Stage 2 — compute DINO / CLIP-I / CLIP-T metrics
!python main.py eval \
 --config=/content/dreambooth-replication/configs/subjects.yaml \
 --stage=metrics \
 --subject=dog \
 --checkpoint-dir="/content/drive/MyDrive/dreambooth/checkpoints" \
 --instance-dir=artifacts/subject_images \
 --output-dir="/content/drive/MyDrive/dreambooth/eval_output" \
 --results-dir="/content/drive/MyDrive/dreambooth/results" \
 --device=cuda
```

eval has two stages:

_generate_ loads the fine-tuned checkpoint for each subject and runs it on 25 prompts (e.g. "a sks dog on the beach"), saving 4 images per prompt. Also generates the same prompts w/o identifier

_metrics_ computes three scores against the original instance images:

- **DINO**: how much the generated images look like the specific subject (identity similarity). Needs generated subjects + original subjects
- **CLIP-I**: similar to DINO. Needs generated subjects + original subjects
- **CLIP-T**: how well the generated images match the text prompt. Needs prompt text + generated subjects

reference images for DINO and CLIP-I are the original training photos of the subject, not the class images. results written to `RESULTSDIR/metrics.json`.
