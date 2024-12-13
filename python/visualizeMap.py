import numpy as np
import matplotlib.pyplot as plt
import skimage.color

'''
genMap = list()
fd = open("/Users/victorz/Desktop/15418-FinalProject/octave.txt", "r")
next(fd) # Skip the first line
for line in fd:
    linearray = [(0.5 * (float(n) + 1)) for n in line.split()]
    genMap.append(linearray)

image_data = np.array(genMap, dtype="float")
plt.imshow(image_data, interpolation='nearest')
plt.axis('off')
plt.savefig("heightmap.png", bbox_inches='tight', pad_inches=0)
plt.show() '''

voronoi = list()
fd = open("/Users/victorz/Desktop/15418-FinalProject/voronoi_biome.txt", "r")
next(fd) # Skip the first line
for line in fd:
    linearray = [n for n in line.split()]
    voronoi.append(linearray)

image_data = np.array(voronoi, dtype="int")

label_image = skimage.color.label2rgb(image_data)
plt.imshow(image_data, interpolation='nearest')
plt.axis('off')
plt.savefig("voronoi2.png", bbox_inches='tight', pad_inches=0)
plt.show()

voronoi = list()
fd = open("/Users/victorz/Desktop/15418-FinalProject/voronoi_distance.txt", "r")
next(fd) # Skip the first line
for line in fd:
    linearray = [n for n in line.split()]
    voronoi.append(linearray)

image_data = np.array(voronoi, dtype="float")
image_data /= np.max(image_data)
image_data = 1. - image_data

label_image = skimage.color.label2rgb(image_data)
plt.imshow(image_data, interpolation='nearest')
plt.axis('off')
plt.savefig("voronoi2dist.png", bbox_inches='tight', pad_inches=0)
plt.show()
