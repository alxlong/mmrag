from mmrag.vectordb import connect, build_table
import glob

db = connect()
embedded_files = sorted(glob.glob('data/embeddings/shard_*.parquet'))

index = build_table(db, embedded_files)
