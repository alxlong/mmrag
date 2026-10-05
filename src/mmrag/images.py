import tarfile, io
from PIL import Image
from pathlib import Path

# needed for functionality in Jupyter notebook
ROOT = Path(__file__).resolve().parents[2]  

def load_image(key, shard_number):
    path = ROOT / f"data/raw/cc12m-train-{shard_number:04d}.tar"

    with tarfile.open(path) as tf:
        f = tf.extractfile(f"{key}.jpg")
        return Image.open(io.BytesIO(f.read()))
