#ifndef __BIOMEMAP_H__
#define __BIOMEMAP_H__

#include "util.h"

// Color Map struct, for use on the CPU
struct BiomeMap {

  BiomeMap(int w, int h) {
    width = w;
    height = h;
    data = new biomeData[width * height];
  }

  void clear() {

    int numPixels = width * height;
    biomeData* ptr = data;
    for (int i = 0; i < numPixels; i++) {
      ptr[0].pixelBiome = BLANK;
      ptr[0].pixelDist = 0.0f;
      ptr += 1;
    }
  }

  int width;
  int height;
  biomeData* data;
};

#endif