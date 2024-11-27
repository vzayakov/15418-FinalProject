// Main CUDA file for our parallel terrain generation algorithm
// Authors: Petros Emmanouilidis and Victor Zayakov

#include <string>
#include <algorithm>
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>
#include <vector>

#include "noiseMap.h"
#include "util.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

// This stores the global constants
struct GlobalConstants {

  int noiseMapWidth;
  int noiseMapHeight;
  float* noiseMapData;

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

}

// Destructor for class
TerrainGen::~TerrainGen() {
  
  if (noiseMap) {
    delete noiseMap;
  }

  if (cudaDeviceNoiseMapData) {
    cudaFree(cudaDeviceNoiseMapData);
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
    *(float*)(&cuConstRendererParams.noiseMapData[offset]) = value;
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

    cudaMemcpyToSymbol(cuConstRendererParams, &params, sizeof(GlobalConstants));

}

// allocOutputNoiseMap --
//
// Allocate buffer where we'll put the noise map on the CPU.  Check status of
// noise map first to avoid memory leak.
void TerrainGen::allocOutputNoiseMap(int width, int height) {

    if (noiseMap)
        delete noiseMap;
    noiseMap = new NoiseMap(width, height);
}

// clearNoiseMap --
//
// Clear the generated noise map.
void TerrainGen::clearNoiseMap() {

    // 256 threads per block is a healthy number
    dim3 blockDim(16, 16, 1);
    dim3 gridDim(
        (noiseMap->width + blockDim.x - 1) / blockDim.x,
        (noiseMap->height + blockDim.y - 1) / blockDim.y);

    kernelClearNoiseMap<<<gridDim, blockDim>>>(1.f, 1.f, 1.f, 1.f);
    cudaDeviceSynchronize();
}


// Kernel that generates the Perlin noise map on the GPU
__global__ void perlin() {

}

// Main function that generates the terrain. Makes all of the necessary
// kernel calls
void TerrainGen::generate() {
  // Call perlin() here, maybe other kernels too
}