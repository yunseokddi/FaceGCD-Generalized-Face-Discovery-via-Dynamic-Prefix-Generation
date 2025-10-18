import os
import numpy as np
import torch

root_dir = 'gcd_youtubefaces_1000_part_fvit_pretrain_prefix10/test'
save_dir = 'gcd_youtubefaces_1000_part_fvit_pretrain_prefix10/npy_test'
features = []
labels = []

for class_name in sorted(os.listdir(root_dir), key=lambda x: int(x)):
    class_path = os.path.join(root_dir, class_name)
    if not os.path.isdir(class_path):
        continue
    label = int(class_name)

    for fname in os.listdir(class_path):
        if fname.endswith('.npy'):
            fpath = os.path.join(class_path, fname)
            feat = torch.load(fpath)
            if feat.ndim == 1:
                feat = feat[np.newaxis, :]
            features.append(feat)
            labels.append(label)

feature_array = np.vstack(features)
label_array = np.array(labels)

np.save(os.path.join(save_dir, 'feature.npy'), feature_array)
np.save(os.path.join(save_dir, 'label.npy'), label_array)

print(f" feature.npy saved: shape = {feature_array.shape}")
print(f" label.npy saved: shape = {label_array.shape}")
