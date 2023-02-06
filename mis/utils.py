import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

plt.style.use("ggplot")


def plot_from_img_path(rows, columns, list_img_path, list_mask_path):
    fig = plt.figure(figsize=(12, 12))
    for i in range(1, rows * columns + 1):
        fig.add_subplot(rows, columns, i)
        img_path = list_img_path[i]
        mask_path = list_mask_path[i]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path)
        # plt.imshow(image)
        plt.imshow(mask)
    plt.show()