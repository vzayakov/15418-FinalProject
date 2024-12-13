#ifndef __UTIL_H__
#define __UTIL_H__

// Header file that contains useful macros
#include <algorithm>
#include <math.h>
#define CLAMP(x, minimum, maximum) fmaxf(minimum, fminf(x, maximum))

// 2D Vector struct
typedef struct {
  float x, y;
} vector2;

// Color enum for color map
typedef enum biome {
  BLANK = -1,
  MOUNTAIN = 0,
  OCEAN = 1,
  DESERT = 2,
  PLAIN = 3,
  HILLS1 = 4,
  HILLS2 = 5,
  HILLS3 = 6
} Biome;

// Voronoi pixel struct, contains biome type and distance to seed point
typedef struct {
  float pixelDist;
  Biome pixelBiome;
} biomeData;

// Voronoi point coordinate struct
typedef struct {
  int x, y;
  Biome b;
} pointCoord;

#endif