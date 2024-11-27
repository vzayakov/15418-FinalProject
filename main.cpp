// 15-418 Final Project: Parallel Terrain Generation
// Authors: Petros Emmanouilidis and Victor Zayakov
// Main C++ file for our final project, parses command line arguments
// and instantiaties the Cuda class

#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>
#include <fstream>
#include <iostream>

#include "terrainGen.h"
#include "noiseMap.h"

// Helper function that writes the generated noise map to a .txt file
void writeNoiseMap(TerrainGen* generator, const int dimX, const int dimY) {

  const NoiseMap* noiseMap = generator->getNoiseMap();
  const std::string outputFilename = "noisemap.txt";

  std::ofstream outNoiseMap(outputFilename, std::fstream::out);
  if (!outNoiseMap) {
    std::cerr << "Unable to open file: " << outputFilename << '\n';
    exit(EXIT_FAILURE);
  }

  outNoiseMap << dimX << ' ' << dimY << '\n';
  for (int i = 0; i < dimY; i++) {
    for (int j = 0; j < dimX; j++) {
      outNoiseMap << noiseMap->data[i * dimX + j] << ' ';
    }
    outNoiseMap << '\n';
  }

  outNoiseMap.close();

}

int main(int argc, char** argv) {
  // Can modify this to have noise maps of different sizes
  int noiseMapWidth = 1150;
  int noiseMapHeight = 1150;
  // Instantiate the class object
  TerrainGen* generator = new TerrainGen();

  // Allocate noise map and set up all of the things
  generator->allocOutputNoiseMap(noiseMapWidth, noiseMapHeight);
  generator->setup();
  // Generate the noise map
  generator->generate();
  // Write the generated noise map to a .txt file
  writeNoiseMap(generator, noiseMapWidth, noiseMapHeight);

  return 0;

}