// Main CUDA file for our parallel terrain generation algorithm
// Authors: Petros Emmanouilidis and Victor Zayakov

#include <string>
#include <algorithm>
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>
#include <vector>



#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

// Constructor for class
TerrainGen::TerrainGen() {
  
}

// Destructor for class
TerrainGen::~TerrainGen() {
  
}

// Add more methods here