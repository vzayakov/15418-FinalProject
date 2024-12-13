#ifndef __COLORMAP_H__
#define __COLORMAP_H__

#include "util.h"

// Color Map struct, for use on the CPU
struct ColorMap {

  ColorMap(int w, int h) {
    width = w;
    height = h;
    data = new Color[width * height];
  }

  void clear() {

    int numPixels = width * height;
    Color* ptr = data;
    for (int i = 0; i < numPixels; i++) {
      ptr[0] = BLANK;
      ptr += 1;
    }
  }

  int width;
  int height;
  Color* data;
};

#endif