import os
import random

import numpy as np
import torch
from skimage.io import imread
from torch.utils.data import Dataset

subset="train"
assert subset in ["all", "train", "validation"]
volumes = {}
masks = {}
images_dir = 'kaggle_3m'
print("reading {} images...".format(subset))
count = 0
for (dirpath, dirnames, filenames) in os.walk(images_dir):
    
    image_slices = []
    mask_slices = []
    for filename in sorted(
        filter(lambda f: ".tif" in f, filenames),
        key=lambda x: int(x.split(".")[-2].split("_")[4]),
    ):
        filepath = os.path.join(dirpath, filename)
        if "mask" in filename:
            mask_slices.append(imread(filepath, as_gray=True))
        else:
            image_slices.append(imread(filepath))
    if len(image_slices) > 0:
        patient_id = dirpath.split("\\")[-1]
        print(f"PATIENT ID = {patient_id}")
        volumes[patient_id] = np.array(image_slices[1:-1])
        masks[patient_id] = np.array(mask_slices[1:-1])
    print(volumes)
    if count ==1:
        break
    count = count+1