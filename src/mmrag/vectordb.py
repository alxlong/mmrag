import lancedb
import pandas as pd

def connect(path='data/lancedb'):
    return lancedb.connect(path)

def build_table(db, embedded_files, name="CLIP_images"):
    table = db.create_table(name, data=pd.read_parquet(embedded_files[0]))

    for embedded_file in embedded_files[1:]:
        table.add(pd.read_parquet(embedded_file))

    return table
