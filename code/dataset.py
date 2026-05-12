from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class DreamBoothDataset(Dataset):
    def __init__(self, instance_dir, instance_prompt, class_dir, class_prompt, tokenizer, size=512):
        self.instance_paths = sorted(Path(instance_dir).glob("*.[jp][pn]g"))
        self.class_paths = sorted(Path(class_dir).glob("*.[jp][pn]g"))
        self.instance_prompt = instance_prompt
        self.class_prompt = class_prompt
        self.tokenizer = tokenizer

        self.transform = T.Compose([
            T.Resize(size, interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop(size),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),  # SD VAE expects [-1, 1]
        ])

    def __len__(self):
        # iterate until the longer list is exhausted; shorter one wraps via modulo
        return max(len(self.instance_paths), len(self.class_paths))

    def _tokenize(self, prompt):
        return self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

    def __getitem__(self, idx):
        instance_image = Image.open(self.instance_paths[idx % len(self.instance_paths)]).convert("RGB")
        class_image = Image.open(self.class_paths[idx % len(self.class_paths)]).convert("RGB")
        return {
            "instance_pixel_values": self.transform(instance_image),
            "instance_input_ids": self._tokenize(self.instance_prompt),
            "class_pixel_values": self.transform(class_image),
            "class_input_ids": self._tokenize(self.class_prompt),
        }
