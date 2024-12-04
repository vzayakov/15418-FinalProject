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
    float* cudaDeviceNoiseMapData; // GPU noise map
    short* cudaDevicePermutationTable; // Permutation table
  
  public:
    // We can add items to the class here: constructor, destructor, methods, etc
    TerrainGen(); // Constructor
    virtual ~TerrainGen(); // Destructor

    const NoiseMap* getNoiseMap(); // Copy noise map from device to CPU

    void setup(); // Initial setup function

    void allocOutputNoiseMap(int width, int height); // Allocate CPU noise map

    void clearNoiseMapDevice(); // Clears the noise map on the device

    void generateSpatial(int initialGridSize, int octaves, int persistence, 
                          int lacunarity); // Run perlin kernel, spatial
    
    void generateTemporal(int initialGridSize, int octaves, int persistence, 
                          int lacunarity); // Run perlin kernel, temporal

};

#endif