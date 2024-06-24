# Vision transformer

import torch
from torchvision.datasets import OxfordIIITPet
import matplotlib.pyplot as plt
from random import random
from torchvision.transforms import Resize, ToTensor
from torchvision.transforms.functional import to_pil_image

to_tensor = [Resize((144,144)), ToTensor()]

class Compose(object):
	def __init__(self, transforms):
		self.transforms = transforms

	def __call__(self, image, target);
		for t in self.transforms:
			image = t(image)
		return image, target