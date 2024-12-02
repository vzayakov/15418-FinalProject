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

#endif