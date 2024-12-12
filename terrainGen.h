// Header file for our CUDA code
// We can use this for class instantiation, constants, function definitions

#ifndef __TERRAIN_GEN_H__
#define __TERRAIN_GEN_H__

#ifndef uint
#define uint unsigned int
#endif

struct NoiseMap;

class TerrainGen {

  private:
  
    NoiseMap* noiseMap; // CPU noise map
    ColorMap* colorMap;
    float* cudaDeviceNoiseMapData; // GPU noise map
    float* cudaDeviceColorMapData; // GPU color map (voronoi)
    float* cudaDevicePartialSums;
    short* cudaDevicePermutationTable; // Permutation table
  
  public:
    // We can add items to the class here: constructor, destructor, methods, etc
    TerrainGen(); // Constructor
    virtual ~TerrainGen(); // Destructor

    const NoiseMap* getNoiseMap(); // Copy noise map from device to CPU

    const ColorMap* getColorMap(); // Copy color map from device to CPU

    void setup(int octaves); // Initial setup function

    void allocOutputColorMap(int width, int height); // Allocate CPU color map

    void allocOutputNoiseMap(int width, int height); // Allocate CPU noise map

    void clearNoiseMapDevice(); // Clears the noise map on the device

    void generateSpatial(int initialGridSize, int octaves, float persistence, 
                         float lacunarity); // Run perlin kernel, spatial
    
    void generateTemporal(int initialGridSize, int octaves, float persistence, 
                          float lacunarity); // Run perlin kernel, temporal
    
    void generateVoronoi(int gridSize); // Run Voronoi noise generation kernel

};

#endif