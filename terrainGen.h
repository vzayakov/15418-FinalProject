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
  
    NoiseMap* noiseMap;
    float* cudaDeviceNoiseMapData;
  
  public:
    // We can add items to the class here: constructor, destructor, methods, etc
    TerrainGen(); // Constructor
    virtual ~TerrainGen(); // Destructor

    const NoiseMap* getNoiseMap();

    void setup();

    void allocOutputNoiseMap(int width, int height);

    void clearNoiseMap();

    void generate();

};

#endif