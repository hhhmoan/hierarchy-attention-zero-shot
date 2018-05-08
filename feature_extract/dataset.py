import os
import numpy as np
from PIL import Image
import csv
import torch
import torch.utils.data as data
import torch.nn.functional as F
import torchvision.transforms as transforms

def load_from_txt(image_root, image_list):
	image_name_list = []
	labels = []
	with open(image_list, 'r') as name_list:
		for i in name_list.readlines():
			order, path = i.split(' ')
			labels.append(int(path[0:3]))
			image_name_list.append(image_root+path[:-1])
	return image_name_list, labels

class CUB_data(data.Dataset):
	def __init__(self, image_root, image_list, img_preprocess):
		super(CUB_data, self).__init__()
		self.image_root = image_root
		self.image_list = image_list
		self.data = self.load_data()
		self.img_preprocess = img_preprocess
	def load_data(self):
		image_name_list, labels = load_from_txt(self.image_root, self.image_list)
		length = len(image_name_list)
		output = [[image_name_list[i], labels[i]] for i in range(length)]
		return output
	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		datam = self.data[index]
		image = Image.open(datam[0])
		if image.mode == 'L':
			image = image.convert('RGB')
		label = datam[1]
		img = self.img_preprocess(image)
		return img, label, datam[0]




