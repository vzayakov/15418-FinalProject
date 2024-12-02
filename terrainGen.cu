// Main CUDA file for our parallel terrain generation algorithm
// Authors: Petros Emmanouilidis and Victor Zayakov

#include <string>
#include <algorithm>
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>
#include <vector>
#include <cstdlib>

#include "noiseMap.h"
#include "util.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

// Permutation table for Perlin noise
short permutationGlobal[256] = { 151,160,137,91,90,15,
   131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,
   190, 6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,
   88,237,149,56,87,174,20,125,136,171,168, 68,175,74,165,71,134,139,48,27,166,
   77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,
   102,143,54, 65,25,63,161, 1,216,80,73,209,76,132,187,208, 89,18,169,200,196,
   135,130,116,188,159,86,164,100,109,198,173,186, 3,64,52,217,226,250,124,123,
   5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,
   223,183,170,213,119,248,152, 2,44,154,163, 70,221,153,101,155,167, 43,172,9,
   129,22,39,253, 19,98,108,110,79,113,224,232,178,185, 112,104,218,246,97,228,
   251,34,242,193,238,210,144,12,191,179,162,241, 81,51,145,235,249,14,239,107,
   49,192,214, 31,181,199,106,157,184, 84,204,176,115,121,50,45,127, 4,150,254,
   138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180
   };

// This stores the global constants
struct GlobalConstants {

  int noiseMapWidth;
  int noiseMapHeight;
  float* noiseMapData;
  short* permutation;

};

// Global variable that is in scope, but read-only, for all cuda
// kernels.  The __constant__ modifier designates this variable will
// be stored in special "constant" memory on the GPU. (we didn't talk
// about this type of memory in class, but constant memory is a fast
// place to put read-only variables).
__constant__ GlobalConstants cuConstTerrainGenParams

// Debugging macro
#define gpuErrChk() { gpuAssert(__FILE__, __LINE__); }
inline void gpuAssert(const char *file, int line, bool abort=true)
{
    cudaError_t code = cudaPeekAtLastError();
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// Constructor for class
TerrainGen::TerrainGen() {

  noiseMap = NULL;
  cudaDeviceNoiseMapData = NULL;
  cudaDevicePermutationTable = NULL;

}

// Destructor for class
TerrainGen::~TerrainGen() {
  
  if (noiseMap) {
    delete noiseMap;
  }

  if (cudaDeviceNoiseMapData) {
    cudaFree(cudaDeviceNoiseMapData);
    cudaFree(cudaDevicePermutationTable);
  }
}

// ADD MORE METHODS HERE

// kernelClearNoiseMap --  (CUDA device code)
//
// Clear the noise map, setting all pixels to the specified color rgba
__global__ void kernelClearNoiseMap(float h) {

    int noiseMapX = blockIdx.x * blockDim.x + threadIdx.x;
    int noiseMapY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstTerrainGenParams.noiseMapWidth;
    int height = cuConstTerranGenParams.noiseMapHeight;

    if (noiseMapX >= width || noiseMapY >= height)
        return;

    int offset = (noiseMapY * width + noiseMapX);
    float value = h;

    // Write to global memory
    *(float*)(&cuConstTerrainGenParams.noiseMapData[offset]) = value;
}

const noiseMap* TerrainGen::getNoiseMap() {

    // Need to copy contents of the generated noiseMap from device memory
    // before we expose the noiseMap object to the caller

    printf("Copying noise map data from device\n");

    cudaMemcpy(noiseMap->data,
               cudaDeviceNoiseMapData,
               sizeof(float) * noiseMap->width * noiseMap->height,
               cudaMemcpyDeviceToHost);

    return noiseMap;
}

void TerrainGen::setup() {

    int deviceCount = 0;
    bool isFastGPU = false;
    std::string name;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Initializing CUDA for TerrainGen\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        name = deviceProps.name;
        if (name.compare("GeForce RTX 2080") == 0)
        {
            isFastGPU = true;
        }

        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
    if (!isFastGPU)
    {
        printf("WARNING: "
               "You're not running on a fast GPU, please consider using "
               "NVIDIA RTX 2080.\n");
        printf("---------------------------------------------------------\n");
    }
    
    // Copy all the key
    // data structures into device memory so they are accessible to
    // CUDA kernels
    //
    // See the CUDA Programmer's Guide for descriptions of
    // cudaMalloc and cudaMemcpy
    cudaMalloc(&cudaDeviceNoiseMapData, sizeof(float) * noiseMap->width * noiseMap->height);
    cudaMalloc(&cudaDevicePermutationTable, sizeof(short) * 256);

    cudaMemcpy(cudaDevicePermutationTable, permutationGlobal,
               sizeof(short) * 256, cudaMemcpyHostToDevice);

    // Initialize parameters in constant memory.  We didn't talk about
    // constant memory in class, but the use of read-only constant
    // memory here is an optimization over just sticking these values
    // in device global memory.  NVIDIA GPUs have a few special tricks
    // for optimizing access to constant memory.  Using global memory
    // here would have worked just as well.  See the Programmer's
    // Guide for more information about constant memory.

    GlobalConstants params;
    params.noiseMapWidth = noiseMap->width;
    params.noiseMapHeight = noiseMap->height;
    params.noiseMapData = cudaDeviceNoiseMapData;
    params.permutation = cudaDevicePermutationTable;

    cudaMemcpyToSymbol(cuConstTerrainGenParams, &params, sizeof(GlobalConstants));

}

// allocOutputNoiseMap --
//
// Allocate buffer where we'll put the noise map on the CPU.  Check status of
// noise map first to avoid memory leak.
// Also sets all squares to 0.
void TerrainGen::allocOutputNoiseMap(int width, int height) {

    if (noiseMap)
        delete noiseMap;
    noiseMap = new NoiseMap(width, height);
    noiseMap.clear(0.f); // Set all squares to 0
}

// clearNoiseMapDevice --
//
// Clear the generated noise map, on the device.
void TerrainGen::clearNoiseMapDevice() {

    // 256 threads per block is a healthy number
    dim3 blockDim(16, 16, 1);
    dim3 gridDim(
        (noiseMap->width + blockDim.x - 1) / blockDim.x,
        (noiseMap->height + blockDim.y - 1) / blockDim.y);

    kernelClearNoiseMap<<<gridDim, blockDim>>>(0.f);
    cudaDeviceSynchronize();
}

// Interpolation, using the smootherstep() function
// first and second derivatives are both 0
__device__ __inline__ float interpolate(float a0, float a1, float w) {

  return (a1 - a0) * (w * w * w * (w * (w * 6 - 15) + 10)) + a0;
}

// Computes the dot product of a pixel's offset vector with a given gradient
__device__ __inline__ float dotGridGradient(int ix, int iy, float x, float y
                      int gridLeftCoord, int gridRightCoord, int gridTopCoord,
                      float* gradients) {

  // Get gradient from integer coordinates
  int index = (iy - gridTopCoord) * (gridRightCoord - gridLeftCoord) + (ix - gridLeftCoord);
  float gridAngle = gradients[index]
  
  vector2 gradient;
  gradient.x = cos(gradAngle);
  gradient.y = sin(gradAngle);

  // Compute the distance vector
  float dx = x - (float)(ix);
  float dy = y - (float)(iy);

  // Compute the dot product
  return (dx * gradient.x + dy * gradient.y);

}

// Kernel that generates the Perlin noise map on the GPU
__global__ void perlin(int noiseMapWidth, int noiseMapHeight,
                       int initialGridSize, int octaves, int persistence,
                       int lacunarity, int blockSize) {

  // get global and local thread indices
  int threadIndex = threadIdx.y * blockDim.x + threadIdx.x;
  int pixelX = blockIdx.x * blockDim.x + threadIdx.x;
  int pixelY = blockIdx.y * blockDim.y + threadIdx.y;

  // Number of threads
  int number_of_threads = blockDim.x * blockDim.y;

  // Initial number of pixel per grid cell
  int gridSize = initialGridSize;

  // Used for computing gradient values
  __shared__ short permutationTable[256];
  // Used for storing intermediate pixel values
  __shared__ float pixelValues[1024];

  // Used for storing gradient values
  extern __shared__ float gradients[];

  // Set up permutation table
  if (threadIndex < 256) {
    permutationTable[threadIndex] = permutation[threadIndex];
  }
  __syncthreads();

  // Local possible gradient values
  float grad_vals [4] = {M_PI / 4.f, 3.f * M_PI / 4.f, 5.f * M_PI / 4.f, 7.f * M_PI / 4.f};

  // For every iteration
  for (int iteration = 0; iteration < octaves; iteration++) {


    // FIRST: SET UP GRADIENTS FOR ITERATION
    ////////////////////////////////////////
    int gridLeftCoord = (blockSize * blockIdx.x) / gridSize;
    int gridRightCoord = ((blockSize * (blockIdx.x + 1)) / gridSize) + 1;
    int gridTopCoord = (blockSize * blockIdx.y) / gridSize;
    int gridBottomCoord = ((blockSize * (blockIdx.y + 1)) / gridSize) + 1;

    // Number of grid cells enclosed by the block
    int gridNumber = (gridRightCoord - gridLeftCoord) * (gridBottomCoord - gridTopCoord); 

    // Simplify things by assuming that each block
    // perfectly encloses a set of square grids

    // Given that assumption, this is the number
    // of intersections / gradients
    int number_of_gradients = (gridRightCoord - gridLeftCoord + 1) * (gridRightCoord - gridLeftCoord + 1);

    // Number of gradients could be larger than number of 
    // of threads in the block
    for (int i = 0; i < number_of_gradients; i += number_of_threads) {

      // Make sure we are within bounds
      if (i * number_of_threads + threadIndex < number_of_gradients) {

        // Local gradient coordinates
        int gradient_block_coord = i * number_of_threads + threadIndex;

        // Global gradient coordinates
        int gradient_globalX = gridLeftCoord + (gradient_block_coord % (gridRightCoord - gridLeftCoord));
        int gradient_globalY = gridTopCoord  + (gradient_block_coord / (gridBottomCoord - gridTopCoord));

        // Grab hash from permutation table
        int hash = permutationTable[(permutationTable[(permutationTable[gradient_globalX % 256] + gradient_globalY) % 256] + iteration) % 256];

        // Use hash to get gradient value
        gradients[gradient_block_coord] = grad_vals[hash % 4];
      }
    }
    __syncthreads();
    ////////////////////////////////////////



    // SECOND: COMPUTE NEW VALUES
    ////////////////////////////////////////
    float value = 0;
    // Given that thread maps to valid pixel
    if ((pixelX < noiseMapWidth) && (pixelY < noiseMapHeight)) {

      // Get local/grid coordinates for pixel gradients
      // We want local coordinates because these are the indices
      // used for grabbing the correct gradients
      
      float left = pixelX / gridSize;
      float top  = pixelY / gridSize;

      // Determine grid cell corner coordinates
      int XLeft = floor(x);
      int YTop = floor(y);
      int XRight = XLeft + 1;
      int YBottom = YTop + 1;

      // Compute interpolation weights
      float sx = x - (float)(XLeft);
      float sy = y - (float)(YTop);

      // Compute and interpolate top two corners
      float n0 = dotGridGradient(XLeft, YTop, x, y, gridLeftCoord, gridRightCoord,
                                 gridTopCoord, &gradients);
      float n1 = dotGridGradient(XRight, YTop, x, y, gridLeftCoord, gridRightCoord,
                                 gridTopCoord, &gradients);
      float ix0 = interpolate(n0, n1, sx);

      // Compute and interpolate bottomn two corners
      n0 = dotGridGradient(XLeft, YBottom, x, y, gridLeftCoord, gridRightCoord,
                           gridTopCoord, &gradients);
      n1 = dotGridGradient(XRight, YBottom, x, y, gridLeftCoord, gridRightCoord,
                           gridTopCoord, &gradients);
      float ix1 = interpolate(n0, n1, sx);

      // Final step: Interpolate between the two resulting values, now in y
      value = interpolate(ix0, ix1, sy);

    }

    ////////////////////////////////////////


    // THIRD: STORE NEW VALUES LOCALLY
    ////////////////////////////////////////

    ////////////////////////////////////////
    pixelValues[threadIndex] = value;
      
  }
  cuConstTerrainGenParams.noiseMapData[pixelY * noiseMapWidth + pixelX];

}

// NOTES
/*
  Grid edges will be located at edges of pixels. We will use pixel centers
  to compute the offset vectors.
*/ 

// Main function that generates the terrain. Makes all of the necessary
// kernel calls
void TerrainGen::generate(int initialGridSize, int octaves, int persistence, 
                          int lacunarity) {
  // Call perlin() here, maybe other kernels too

  int noiseMapWidth = cuConstTerrainGenParams.noiseMapWidth;
  int noiseMapHeight = cuConstTerrainGenParams.noiseMapHeight;

  const int threadX = 32;
  const int threadY = 32;
  const int blockSize = 32;
  const int blockX = (noiseMapWidth + threadX - 1) / threadX;
  const int blockY = (noiseMapHeight + threadY - 1) / threadY;

  dim3 threadsPerBlock(threadX, threadY, 1);
  dim3 numBlocks(blockX, blockY, 1);
  perlin <<numBlocks, threadsPerBlock,  (blockSize + 1) * (blockSize + 1)>> (noiseMapWidth, noiseMapHeight,
                                                                             initialGridSize, octaves, persistence, lacunarity,
                                                                             blockSize);
}