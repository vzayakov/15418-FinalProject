// 15-418 Final Project: Parallel Terrain Generation
// Authors: Petros Emmanouilidis and Victor Zayakov
// Main C++ file for our final project, parses command line arguments
// and instantiaties the Cuda class

#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>

#include "terrainGen.h"
#include "noiseMap.h"

void startTerrainGen(TerrainGen* generator) {

  const NoiseMap* noiseMap = generator->getNoiseMap();

}

int main(int argc, char** argv) {

  int imageSize = 1150;

  TerrainGen* generator = new TerrainGen();

  generator->allocOutputNoiseMap(imageSize, imageSize);
  generator->setup();

  startTerrainGen(generator);

  return 0;

}