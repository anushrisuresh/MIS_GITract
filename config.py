# import the necessary packages
import os
import torch

# base path of the dataset
dataset_path = os.path.join("dataset", "train")

# define the path to the images and masks dataset
image_dataset_path = os.path.join(dataset_path, "images")
mask_dataset_path = os.path.join(dataset_path, "masks")

# define the test split
test_split = 0.15

# determine the device to be used for training and evaluation
device = "cuda" if torch.cuda.is_available() else "cpu"

# determine if we will be pinning memory during data loading
pin_memory = True if device == "cuda" else False

# define the number of channels in the input, number of classes,
# and number of levels in the U-Net model
num_channels = 1
num_classes = 1
num_levels = 3
# initialize learning rate, number of epochs to train for, and the
# batch size
init_lr = 0.001
num_epochs = 40
batch_size = 64
# define the input image dimensions
input_image_width = 128
input_image_height = 128
# define threshold to filter weak predictions
threshold = 0.5
# define the path to the base output directory
base_output = "output"
# define the path to the output serialized model, model training
# plot, and testing image paths
model_path = os.path.join(base_output, "unet_tgs_salt.pth")
plot_path = os.path.sep.join([base_output, "plot.png"])
test_pathS = os.path.sep.join([base_output, "test_paths.txt"])