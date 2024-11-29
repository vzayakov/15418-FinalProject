#ifndef __REF_SERIAL_H__
#define __REF_SERIAL_H__

  struct NoiseMap;
  struct vector2;

  float interpolate(float a0, float a1, float w);

  float perlinRef(float x, float y);

  vector2 randomGradient(int ix, int iy);

  float dotGridGradient(int ix, int iy, float x, float y);

  NoiseMap* refSerialMain(int scale, int persistence, int lacunarity, int octaves);

#endif