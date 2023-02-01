# import the necessary packages
from torch.utils.data import Dataset
import cv2

class SegmentationDataset(Dataset):
	def __init__(self, image_paths, mask_path, transforms):
		# store the image and mask filepaths, and augmentation
		# transforms
		self.image_paths = image_paths
		self.mask_path = mask_path
		self.transforms = transforms
	def __len__(self):
		# return the number of total samples contained in the dataset
		return len(self.image_paths)
	def __getitem__(self, idx):
		# grab the image path from the current index
		image_path = self.image_paths[idx]
		# load the image from disk, swap its channels from BGR to RGB,
		# and read the associated mask from disk in grayscale mode
		image = cv2.imread(image_path)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		mask = cv2.imread(self.mask_path[idx], 0)
		# check to see if we are applying any transformations
		if self.transforms is not None:
			# apply the transformations to both image and its mask
			image = self.transforms(image)
			mask = self.transforms(mask)
		# return a tuple of the image and its mask
		return (image, mask)