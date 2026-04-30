from datetime import datetime
from pathlib import Path
from PIL import Image

ARTIFACTS_DIR = Path("artifacts")


def save_image(pathname: str, filename: str, data: Image.Image) -> Path:
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    dest = ARTIFACTS_DIR / pathname / f"{filename}_{timestamp}.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    data.save(dest)
    return dest
