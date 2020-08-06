import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.utils.data as data_utils
from torch.autograd import Variable
import math
import torchvision.models as models
from PIL import Image
from deepfool import deepfool
import os
import scipy.spatial.distance as dist
from scipy import spatial
import cv2
import time 


def pre_process_origimage(orig_image):

	im_orig = Image.open(orig_image)

	mean = [ 0.485, 0.456, 0.406 ]
	std = [ 0.229, 0.224, 0.225 ]

	# Remove the mean
	im = transforms.Compose([
	    transforms.Scale(256),
	    transforms.CenterCrop(224),
	    transforms.ToTensor(),
	    transforms.Normalize(mean = mean,
	                         std = std)])(im_orig)

	return im


def clip_tensor(A, minv, maxv):
    A = torch.max(A, minv*torch.ones(A.shape))
    A = torch.min(A, maxv*torch.ones(A.shape))
    return A


def process_pert_image(pert_image):

	mean = [ 0.485, 0.456, 0.406 ]
	std = [ 0.229, 0.224, 0.225 ]
	
	clip = lambda x: clip_tensor(x, 0, 255)

	tf = transforms.Compose([transforms.Normalize(mean=[0, 0, 0], std=list(map(lambda x: 1 / x, std))),
	transforms.Normalize(mean=list(map(lambda x: -x, mean)), std=[1, 1, 1]),
	transforms.Lambda(clip),
	transforms.ToPILImage(),
	transforms.CenterCrop(224)])

	return tf(pert_image.cpu()[0])

def get_labels(label_orig, label_pert):

	labels = open(os.path.join('synset_words.txt'), 'r').read().split('\n')

	str_label_orig = labels[np.int(label_orig)].split(',')[0]
	str_label_pert = labels[np.int(label_pert)].split(',')[0]

	print("Original label: ", str_label_orig)
	print("Perturbed label: ", str_label_pert)

	# print("Original label = ", label_orig)
	# print("Perturbed label = ", label_pert)


def calculate_cossim(orig,pert):

	original_image = orig.numpy().flatten()
	perturbed_image = np.array(pert).flatten()

	dot_product = np.dot(original_image, perturbed_image)
	norm_a = np.linalg.norm(original_image)
	norm_b = np.linalg.norm(perturbed_image)
	cossimi = ((dot_product / (norm_a * norm_b)))

	return cossimi



if __name__ == "__main__":

	# retrieve pre trained model and set to eval mode
	net = models.vgg19(pretrained=True)
	net.eval()

	# preprocess original image
	im = pre_process_origimage("test_im2.jpg")

	# create perturbed image
	r, loop_i, label_orig, label_pert, pert_image = deepfool(im, net)

	# print original and predicted labels
	get_labels(label_orig, label_pert)

	# process perturbed image
	final_pert_image = process_pert_image(pert_image)

	# save image
	final_pert_image.save("pert_image.jpg")

	# compute cosine similarity between images
	im2 = Image.open("pert_image.jpg")
	print ("Cosine Similarity:",calculate_cossim(im,im2))












