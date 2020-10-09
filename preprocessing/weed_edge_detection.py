# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 14:32:33 2020

@author: OWNER
"""

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img1 = cv.imread('Weed1.jpeg')
# edges = cv.Canny(img,60,100)
# plt.subplot(121),plt.imshow(img,cmap = 'gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(edges,cmap = 'gray')
# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
# plt.show()

green_channel = img1[:,:,1]

# create empty image with same shape as that of src image
green_img = np.zeros((img1.shape[0],img1.shape[1]))

#assign the green channel of src to empty image
green_img = green_channel

# plt.imshow(green_img)

#plot edges for whole image to compare later
edges = cv.Canny(green_img,150,255)
plt.subplot(121),plt.imshow(green_img)
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()

# Set tile size to be Mt x Nt
# Should divide M,N evenly
Mt = 512
Nt = 512

# Run tiling code
img = green_img
M,N = img.shape
rows = M//Mt
cols = N//Nt

tiles = [img[x:x+Mt,y:y+Nt] for x in range(0,img.shape[0],Mt) for y in range(0,img.shape[1],Nt)]
print("Number of tiles: ", len(tiles))

# Rescale each tile to be the desired image size for the CNN
# CNN takes in (Md x Nd) image
Md = 224
Nd = 224

dim_nn = (Md, Nd)
tiles_resized = [cv.resize(tile, dim_nn, interpolation = cv.INTER_AREA) for tile in tiles]

#Check edge detection for resized tile    
img = tiles_resized[10]

#Auto-set upper and lower bounds for edge detection based on median value
sigma = .5
v = np.median(img)
l = int(max(0,  v)) #Normally we would do (1-sigma)*v but becasue the data isn't normalized? it was not working as well
u = int(min(255, (1.0 + sigma) * v))

edges = cv.Canny(img,l,u)
plt.subplot(121),plt.imshow(img)
plt.title('Tile Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()
