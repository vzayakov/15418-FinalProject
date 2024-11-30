#include <iostream>
#include <cmath>
#include <cstdlib>

#include "noiseMap.h"
#include "util.h"

float interpolate(float a0, float a1, float w) {

  return (a1 - a0) * (w * w * w * (w * (w * 6 - 15) + 10)) + a0;
}

// Sample Perlin noise at coordinates (x, y)
// SOURCE: https://www.youtube.com/watch?v=kCIaHqb60Cw
float perlinRef(float x, float y) {

  // Determine grid cell corner coordinates
  int XLeft = floor(x);
  int YTop = floor(y);
  int XRight = XLeft + 1;
  int YBottom = YTop + 1;

  // Compute interpolation weights
  float sx = x - (float)(XLeft);
  float sy = y - (float)(YTop);

  // Compute and interpolate top two corners
  float n0 = dotGridGradient(XLeft, YTop, x, y);
  float n1 = dotGridGradient(XRight, YTop, x, y);
  float ix0 = interpolate(n0, n1, sx);

  // Compute and interpolate bottomn two corners
  n0 = dotGridGradient(XLeft, YBottom, x, y);
  n1 = dotGridGradient(XRight, YBottom, x, y);
  float ix1 = interpolate(n0, n1, sx);

  // Final step: Interpolate between the two resulting values, now in y
  float value = interpolate(ix0, ix1, sy);

  return value;
}

vector2 randomGradient(int ix, int iy) {

  srand(ix * iy); // Seed the RNG
  // Generate a random real number in [0, 1]
  float randomRealNumber = (float)(rand() / RAND_MAX);
  // Convert to range [0, 2pi]
  float randomAngle = randomRealNumber * 3.14159265 * 2.0f;

  // Create the vector from the angle
  vector2 v;
  v.x = cos(randomAngle);
  v.y = sin(randomAngle);

  return v;
}

float dotGridGradient(int ix, int iy, float x, float y) {

  // Get gradient from integer coordinates
  vector2 gradient = randomGradient(ix, iy);

  // Compute the distance vector
  float dx = x - (float)(ix);
  float dy = y - (float)(iy);

  // Compute the dot product
  return (dx * gradient.x + dy * gradient.y);
}

// Main function for serial reference implementation
// SOURCE: https://www.youtube.com/watch?v=kCIaHqb60Cw
NoiseMap* refSerialMain(int scale, int persistence, int lacunarity, int octaves) {

  const int noiseMapWidth = 1150;
  const int noiseMapHeight = 1150;

  NoiseMap* noiseMap;
  noiseMap = new NoiseMap(noiseMapWidth, noiseMapHeight);
  noiseMap.clear(0.f);

  for (int x = 0; x < noiseMapWidth; x++) {

    for (int y = 0; y < noiseMapHeight; y++) {

      int index = (y * noiseMapWidth + x);

      float val = 0;
      float freq = 1;
      float amp = 1;

      for (int i = 0; i < octaves; i++) {
        val += perlinRef(x * freq / scale, y * freq / scale) * amp;

        freq *= lacunarity;
        amp *= persistence; 
      }

      // Clamping to [-1, 1]
      val = CLAMP(val, -1.0f, 1.0f);
      // Set value in noiseMap
      noiseMap->data[index] = val;
    }

  }

}