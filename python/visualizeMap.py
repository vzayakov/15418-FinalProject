import numpy as np
import matplotlib.pyplot as plt

genMap = list()
fd = open("/Users/Yannis/Desktop/15418/15418-FinalProject/test1.txt", "r")
next(fd) # Skip the first line
for line in fd:
    linearray = [(0.5 * (float(n) + 1)) for n in line.split()]
    genMap.append(linearray)

image_data = np.array(genMap, dtype="float")
plt.imshow(image_data, interpolation='nearest')
plt.show()