#ifndef __COLORMAP_H__
#define __COLORMAP_H__

#include "util.h"

// Color Map struct, for use on the CPU
struct ColorMap {

  ColorMap(int w, int h) {
    width = w;
    height = h;
    data = new color[width * height];
  }

  void clear() {

    int numPixels = width * height;
    color* ptr = data;
    for (int i = 0; i < numPixels; i++) {
      ptr[0] = BLANK;
      ptr += 1;
    }
  }

  int width;
  int height;
  color* data;
};

#endif