from huggingface_hub import hf_hub_download
import os

for i in range(0, 15):
    hf_hub_download(local_dir_use_symlinks=False, repo_id="laion/laion-coco", \
                    filename=f"part-0000{i}-2256f782-126f-4dc6-b9c6-e6757637749d-c000.snappy.parquet", \
                    repo_type="dataset", local_dir="./meta")
