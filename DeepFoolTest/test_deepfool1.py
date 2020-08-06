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

net = models.vgg19(pretrained=True)

# net = torch.load_state_dict("mnist-b07bb66b.pth",map_location=torch.device('cpu'))

# Switch to evaluation mode
net.eval()

im_orig = Image.open('test_im2.jpg')

mean = [ 0.485, 0.456, 0.406 ]
std = [ 0.229, 0.224, 0.225 ]


# Remove the mean
im = transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean = mean,
                         std = std)])(im_orig)

start = time.time()

r, loop_i, label_orig, label_pert, pert_image = deepfool(im, net)

end = time.time()

print ("TIME - ",end-start)

# print ("minimal perturbation - ",r)

print ("num_iter - ",loop_i)



print("Original label = ", label_orig)
print("Perturbed label = ", label_pert)



labels = open(os.path.join('synset_words.txt'), 'r').read().split('\n')

str_label_orig = labels[np.int(label_orig)].split(',')[0]
str_label_pert = labels[np.int(label_pert)].split(',')[0]

print("Original label = ", str_label_orig)
print("Perturbed label = ", str_label_pert)

def clip_tensor(A, minv, maxv):
    A = torch.max(A, minv*torch.ones(A.shape))
    A = torch.min(A, maxv*torch.ones(A.shape))
    return A

clip = lambda x: clip_tensor(x, 0, 255)

tf = transforms.Compose([transforms.Normalize(mean=[0, 0, 0], std=list(map(lambda x: 1 / x, std))),
transforms.Normalize(mean=list(map(lambda x: -x, mean)), std=[1, 1, 1]),
transforms.Lambda(clip),
transforms.ToPILImage(),
transforms.CenterCrop(224)])

# print (tf(pert_image.cpu()[0]))

save_image = tf(pert_image.cpu()[0])

# print (save_image.shape)
# print (im.shape)

save_image.save("perturbed_image.jpg")

# im1 = cv2.imread("test_im2.jpg")
im2 = Image.open("perturbed_image.jpg")

original_image = im.numpy().flatten()
perturbed_image = np.array(im2).flatten()


print (original_image.shape)
print (perturbed_image.shape)


dot_product = np.dot(original_image, perturbed_image)
norm_a = np.linalg.norm(original_image)
norm_b = np.linalg.norm(perturbed_image)
cossimi = ((dot_product / (norm_a * norm_b)))

print (cossimi)


# print (im.shape)

# print (spatial.distance.cosine(im.flatten(), pix.flatten()))

# plt.figure()
# plt.imshow(tf(pert_image.cpu()[0]))
# plt.title(str_label_pert)
# plt.show()
