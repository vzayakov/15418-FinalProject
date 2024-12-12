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
enum color {
  BLANK = -1,
  RED = 0,
  GREEN = 1,
  BLUE = 2,
  ORANGE = 3,
  YELLOW = 4,
  PURPLE = 5,
  WHITE = 6
};

// Voronoi point coordinate struct
typedef struct {
  int x, y;
  color clr;
} pointCoord;

#endif