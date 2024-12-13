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
#include <chrono>


#include "terrainGen.h"
#include "refSerial.h"


// Help message
void usage(const char* progname) {
    printf("Usage: %s [options]\n", progname);
    printf("Program Options:\n");
    printf("  -h  --height <INT>         Height of the output noise map, in pixels\n");
    printf("  -w  --width <INT>          Width of the output noise map, in pixels\n");
    printf("  -s  --scale <INT>          Initial grid size, in pixels\n");
    printf("  -o  --octaves <INT>        Number of iterations of Perlin noise\n");
    printf("  -p  --persistence <INT>    Amplitude decay factor over iterations (>1 = increase, <1 = decrease)\n");
    printf("  -l  --lacunarity <INT>     Grid size decrease factor over iterations (>1 = decrease, <1 = increase)\n");
    printf("  -f  --filename <file>      Filepath of output noise map (.txt)\n");
    printf("  -?  --help                 This message\n");
}

// Helper function that writes the generated noise map to a .txt file
void writeNoiseMap(TerrainGen* generator, const int dimX, const int dimY,
                   std::string outputFilename) {

  const NoiseMap* noiseMap = generator->getNoiseMap();

  std::ofstream outNoiseMap(outputFilename, std::fstream::out);
  if (!outNoiseMap) {
    std::cerr << "Unable to open file: " << outputFilename << '\n';
    exit(EXIT_FAILURE);
  }

  outNoiseMap << dimX << ' ' << dimY << '\n'; // Write dimensions at top of file
  for (int i = 0; i < dimY; i++) {
    for (int j = 0; j < dimX; j++) {
      outNoiseMap << noiseMap->data[i * dimX + j] << ' ';
    }
    outNoiseMap << '\n';
  }

  outNoiseMap.close();

}


void writeVoronoi(TerrainGen* generator, const int dimX, const int dimY,
                  std::string outputFilename) {

  const BiomeMap* biomeMap = generator->getBiomeMap();

  std::ofstream outBiomeMap(outputFilename + "_biome.txt", std::fstream::out);
  if (!outBiomeMap) {
    std::cerr << "Unable to open file: " << (outputFilename + "_biome.txt") << '\n';
    exit(EXIT_FAILURE);
  }

  outBiomeMap << dimX << ' ' << dimY << '\n'; // Write dimensions at top of file
  for (int i = 0; i < dimY; i++) {
    for (int j = 0; j < dimX; j++) {
      outBiomeMap << (biomeMap->data[i * dimX + j].pixelBiome) << ' ';
    }
    outBiomeMap << '\n';
  }

  outBiomeMap.close();

  std::ofstream outDistMap(outputFilename + "_distance.txt", std::fstream::out);
  if (!outDistMap) {
    std::cerr << "Unable to open file: " << (outputFilename + "_distance.txt") << '\n';
    exit(EXIT_FAILURE);
  }

  outDistMap << dimX << ' ' << dimY << '\n'; // Write dimensions at top of file
  for (int i = 0; i < dimY; i++) {
    for (int j = 0; j < dimX; j++) {
      outDistMap << (biomeMap->data[i * dimX + j].pixelDist) << ' ';
    }
    outDistMap << '\n';
  }

  outDistMap.close();

}

int main(int argc, char** argv) {

  // Can modify this to have noise maps of different sizes
  int noiseMapWidth = 1150;
  int noiseMapHeight = 1150;
  // Additional input parameters
  int scale = 300;
  float persistence = 1;
  float lacunarity = 1;
  int octaves = 1;
  std::string outputFilename = "noisemap.txt";

  // parse commandline options ////////////////////////////////////////////
  int opt;
  static struct option long_options[] = {
      {"help",     0, 0,  '?'},
      {"height",    0, 0,  'h'},
      {"width",    1, 0,  'w'},
      {"scale",     1, 0,  's'},
      {"octaves", 1, 0,  'o'},
      {"persistence",     1, 0,  'p'},
      {"lacunarity", 1, 0, 'l'},
      {"filename", 0, 0, 'f'},
      {0 ,0, 0, 0}
  };

  // Switch statement for command line arguments
  while ((opt = getopt_long(argc, argv, "f:h:w:s:o:p:l:?", long_options, NULL)) != EOF) {
      switch (opt) {
      case 'f':
        outputFilename = optarg;
        break;
      case 'h':
        noiseMapHeight = atoi(optarg);
        break;
      case 'w':
        noiseMapWidth = atoi(optarg);
        break;
      case 's':
        scale = atoi(optarg);
        break;
      case 'o':
        octaves = atoi(optarg);
        break;
      case 'p':
        persistence = atof(optarg);
        break;
      case 'l':
        lacunarity = atof(optarg);
        break;
      case '?':
      default:
        usage(argv[0]);
        return 1;
      }
  }


  if (optind > argc) {
      fprintf(stderr, "Expected argument after options\n");
      exit(EXIT_FAILURE);
  } 

  // Instantiate the class object
  TerrainGen* generator = new TerrainGen();


  // Allocate noise map and set up all of the things
  generator->allocOutputNoiseMap(noiseMapWidth, noiseMapHeight);
  generator->allocOutputBiomeMap(noiseMapWidth, noiseMapHeight);
  generator->setup(octaves);

  // Taking the average of 100 temporal runs
  double temporalTimes[100]; // Array for storing runtimes of temporal runs

  for (int i = 0; i < 100; i++) {

    // Generate the noise map, using Temporal partitioning
    const auto compute_start_tp = std::chrono::steady_clock::now();
    generator->generateTemporal(scale, octaves, persistence, lacunarity);
    const double compute_time_tp = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - compute_start_tp).count();
    temporalTimes[i] = compute_time_tp;
  }
  
  double temporalAverageTime = 0;
  for (int i = 0; i < 100; i++) {
    temporalAverageTime += temporalTimes[i];
  }
  temporalAverageTime /= 100.0;
  std::cout << "Average Temporal Computation Time (milliseconds): " << (temporalAverageTime * 1000.0) << '\n';

  // generator->setup(octaves);
  
  // Taking the average of 100 spatial runs
  double spatialTimes[100]; // Array for storing runtimes of spatial runs

  for (int i = 0; i < 100; i++) {

    const auto compute_start_sp = std::chrono::steady_clock::now();
    generator->generateSpatial(scale, octaves, persistence, lacunarity);
    const double compute_time_sp = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - compute_start_sp).count();
    spatialTimes[i] = compute_time_sp;

  }

  double spatialAverageTime = 0;
  for (int i = 0; i < 100; i++) {
    spatialAverageTime += spatialTimes[i];
  }
  spatialAverageTime /= 100.0;
  std::cout << "Average Spatial Computation Time (milliseconds): " << (spatialAverageTime * 1000.0) << '\n';

  std::cout << "Voronoi Experiment\n";
  generator->generateVoronoi(300);
  // const auto serial_start = std::chrono::steady_clock::now();
  // NoiseMap * noise_serial = refSerialMain(scale, persistence, lacunarity, octaves);
  // const double serial_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - serial_start).count();
  // std::cout << "Serial Computation time (sec): " << serial_time << '\n';
  // std::cout << "Speedup: " << serial_time / compute_time << "x\n";

  // Write the generated noise map to a .txt file
  writeNoiseMap(generator, noiseMapWidth, noiseMapHeight, outputFilename);

  writeVoronoi(generator, noiseMapWidth, noiseMapHeight, "voronoi");

  delete generator;
  return 0;

}

