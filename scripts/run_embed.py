from mmrag.embed_shard import embed_shard
from pathlib import Path
import pandas as pd
import glob

Path('data/embeddings').mkdir(parents=True, exist_ok=True)

tar_files = sorted(glob.glob('data/raw/cc12m-train-*.tar'))

for tar in tar_files:
    curr_shard_number = int(Path(tar).stem.split("-")[-1])
    curr_shard = Path(f"data/embeddings/shard_{curr_shard_number:04d}.parquet")

    if curr_shard.exists():
        continue

    try:
        vectors, captions, keys = embed_shard(f'data/raw/cc12m-train-{curr_shard_number:04d}.tar')
    except Exception as e:
        print(f'shard {curr_shard_number} failed with exception {e}', flush=True)
        continue

    df = pd.DataFrame({'vector': vectors,
                       'caption': captions,
                       'key': keys,
                       'shard_number': curr_shard_number})

    df.to_parquet(curr_shard) 

    print(f'shard {curr_shard_number} successfully embedded', flush=True)
