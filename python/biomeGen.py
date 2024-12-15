import numpy as np
import matplotlib.pyplot as plt
from skimage.color import *

# DEFINE CONSTANTS
MOUNTAIN_OFFSET = .3
MOUNTAIN_SCALE  = 5.
OCEAN_SCALE     = -5.
OCEAN_CAP       = 0
DESERT_OFFSET   = .2
DESERT_SCALE    = 1.5
PLAIN_OFFSET   = .2
PLAIN_CAP      = 5.
HILL_OFFSET    = .22
HILL_SCALE     = 2.0
DIRT_OFFSET    = .22
DIRT_SCALE     = 1.2
MESSA_SCALE    = 1.5
MESSA_OFFSET   = 3.
MESSA_CAP      = 15.

mountain_count = 0
desert_count = 0
plain_count = 0
hill_count = 0
dirt_count = 0
messa_count = 0

mountain_heights = list()
desert_heights = list()
plain_heights = list()
hill_heights = list()
dirt_heights = list()
messa_heights = list()

# COLORS:
SNOW = [255, 250, 250]
STONE = [105,105,105]
DIRT = [131,101,57]
GRASS_COLD = [37,60,28]
GRASS_WARM = [58,94,44]
GRASS_DRY = [109, 117, 59]
# SAND = [245,235,216]
SAND = [194, 178, 128]	
SAND_TOP = [184,115,51]
CLAY_LIGHT = [224, 212, 182]
CLAY_DARK = [143,75,65]
MUD = [96,70,15]



hMap = list()
vMap = list()
dMap = list()

fd = open("/Users/Yannis/Desktop/15418/15418-FinalProject/test.txt", "r")
next(fd)
for line in fd:
    linearray = [n for n in line.split()]
    hMap.append(linearray)
hMap = np.array(hMap, dtype="float")

fd = open("/Users/Yannis/Desktop/15418/15418-FinalProject/voronoi_biome.txt", "r")
next(fd)
for line in fd:
    linearray = [n for n in line.split()]
    vMap.append(linearray)
vMap = np.array(vMap, dtype="int")

fd = open("/Users/Yannis/Desktop/15418/15418-FinalProject/voronoi_distance.txt", "r")
next(fd)
for line in fd:
    linearray = [n for n in line.split()]
    dMap.append(linearray)
dMap = np.array(dMap, dtype="float")

dMap /= np.max(dMap)
dMap = 1. - dMap


h, w = hMap.shape
biomed_hMap = np.zeros((h, w), dtype="float")
color_map   = np.zeros((h, w, 3), dtype="int")

for i in range(h):
    for j in range(w):

        height = hMap[i,j] + 1.
        biome  = vMap[i, j]
        dist   = dMap[i, j]
        new_height = 0.


        if (biome == 0): 
            new_height = max((MOUNTAIN_OFFSET * dist + dist**3 * height * MOUNTAIN_SCALE), OCEAN_CAP)

            mountain_count+=1
            mountain_heights.append(new_height)

        elif (biome == 1):  
            new_height = min(OCEAN_SCALE * dist * height, OCEAN_CAP)
         

        elif (biome == 2):  
            new_height = max((DESERT_OFFSET + dist * height * DESERT_SCALE), OCEAN_CAP)

            desert_count+=1
            desert_heights.append(new_height)

        elif (biome == 3):  
            new_height = max(min(PLAIN_OFFSET + height, PLAIN_CAP), OCEAN_CAP)

            plain_count+=1
            plain_heights.append(new_height)

        elif (biome == 4):  
            new_height = max((HILL_OFFSET + dist * height * HILL_SCALE), OCEAN_CAP)

            hill_count+=1
            hill_heights.append(new_height)

        elif (biome == 5):  
            new_height = max((DIRT_OFFSET * dist + height * DIRT_SCALE), OCEAN_CAP)

            dirt_count+=1
            dirt_heights.append(new_height)

        elif (biome == 6):  
            new_height = max((min(MESSA_OFFSET * dist + height * MESSA_SCALE, MESSA_CAP)), OCEAN_CAP)

            messa_count+=1
            messa_heights.append(new_height)

        else: print("THIS SHOULD NEVER RUN!!!")

        biomed_hMap[i,j] = new_height

# Max and min values of heightmap
hmax, hmin = float(np.ceil(np.max(biomed_hMap))), float(np.floor(np.min(biomed_hMap)))


mountain_heights.sort()
desert_heights.sort()
plain_heights.sort()
hill_heights.sort()
dirt_heights.sort()
messa_heights.sort()

mountain_heights = np.array(mountain_heights)
desert_heights = np.array(desert_heights)
plain_heights = np.array(plain_heights)
hill_heights = np.array(hill_heights)
dirt_heights = np.array(dirt_heights)
messa_heights = np.array(messa_heights)

# Make height range positive
if hmin < 0.: 
    biomed_hMap += float(np.abs(hmin))
    mountain_heights += float(np.abs(hmin))
    desert_heights += float(np.abs(hmin))
    plain_heights += float(np.abs(hmin))
    hill_heights += float(np.abs(hmin))
    dirt_heights += float(np.abs(hmin))
    messa_heights += float(np.abs(hmin))


# Normalize heightmap
biomed_hMap_norm = biomed_hMap / float(hmax - hmin)

plt.imshow(biomed_hMap_norm, interpolation='nearest')
plt.axis('off')
plt.savefig("biomeheightmap.png", bbox_inches='tight', pad_inches=0)
plt.show()


# Define color cuttoffs for each biome

# Mountain
snow_cuttoff = mountain_heights[int(0.9 * float(mountain_count))]
stone_cuttoff = mountain_heights[int(0.5 * float(mountain_count))]
dirt_cuttoff  = mountain_heights[int(0.3 * float(mountain_count))]

# Desert
desert_top_cuttoff = desert_heights[int(0.9 * float(desert_count))]

# Plain
plain_top_cuttoff = plain_heights[int(0.7 * float(plain_count))]

# Hill
hill_top_cuttoff = hill_heights[int(0.5 * float(hill_count))]

# Dirt
dirt_grass_cuttoff = dirt_heights[int(0.2 * float(dirt_count))]
dirt_mud_cuttoff = dirt_heights[int(0.6 * float(dirt_count))]

# Messa
messa_base_cuttoff = messa_heights[int(0.3 * float(messa_count))]
messa_mid_cuttoff = messa_heights[int(0.7 * float(messa_count))]


for i in range(h):
    for j in range(w):

        height = biomed_hMap[i,j]
        biome  = vMap[i, j]
        color = SNOW

        if (biome == 0):
            if (height < dirt_cuttoff): color = GRASS_COLD
            elif (height < stone_cuttoff): color = DIRT
            elif (height < snow_cuttoff): color = STONE
            else: color = SNOW

        elif (biome == 1):
            color = SAND


        elif (biome == 2):
            if (height < desert_top_cuttoff): color = SAND
            else: color = SAND_TOP

        elif (biome == 3):
            if (height < plain_top_cuttoff): color = GRASS_WARM
            else: color = GRASS_COLD

        elif (biome == 4):
            if (height < hill_top_cuttoff): color = GRASS_WARM
            else: color = DIRT

        elif (biome == 5):
            if (height < dirt_grass_cuttoff): color = MUD
            elif (height < dirt_mud_cuttoff): color = GRASS_DRY
            else: color = GRASS_WARM

        elif (biome == 6):
            if (height < messa_base_cuttoff): color = CLAY_DARK
            elif (height < messa_mid_cuttoff): color = CLAY_LIGHT
            else: color = SAND_TOP

        else: print("This also shouldn't RUN!!!")

        color_map[i,j] = color
        
plt.imshow(color_map, interpolation='nearest')
plt.axis('off')
plt.savefig("colormap.png", bbox_inches='tight', pad_inches=0)
plt.show()

