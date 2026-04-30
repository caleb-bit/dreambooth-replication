# dreambooth-replication

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

Skips subjects whose checkpoint already exists in `OUTPUTDIR/`, so it's safe to rerun after a crash.

```
python main.py eval \
 --config=CONFIGPATH \
 --stage=generate|metrics \
 --subject=SUBJECT \
 --checkpoint-dir=CHECKPOINTDIR \
 --instance-dir=INSTANCEDIR \
 --output-dir=OUTPUTDIR \
 --results-dir=RESULTSDIR \
 --device=DEVICE
```

eval has two stages:

_generate_ loads the fine-tuned checkpoint for each subject and runs it on 25 prompts (e.g. "a sks dog on the beach"), saving 4 images per prompt. Also generates the same prompts w/o identifier

_metrics_ computes three scores against the original instance images:

- **DINO**: how much the generated images look like the specific subject (identity similarity). Needs generated subjects + original subjects
- **CLIP-I**: similar to DINO. Needs generated subjects + original subjects
- **CLIP-T**: how well the generated images match the text prompt. Needs prompt text + generated subjects

reference images for DINO and CLIP-I are the original training photos of the subject, not the class images. results written to `RESULTSDIR/metrics.json`.
