import os
import shutil

items = 'cd_youtubefaces_1000_part_fvit_pretrain_prefix10/train'

for i in range(500, 1000):
    keyword = str(i)
    path = os.path.join(items, keyword)
    try:
        shutil.rmtree(path)
        print(f"Deleted folder: {path}")
    except Exception as e:
        print(f"Failed to delete folder {path}: {e}")