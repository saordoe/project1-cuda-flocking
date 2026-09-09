#define GLM_FORCE_CUDA

#include <cuda.h>
#include "kernel.h"
#include "utilityCore.hpp"

#include <cmath>
#include <cstdio>
#include <iostream>
#include <vector>

#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/device_vector.h>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>

// LOOK-2.1 potentially useful for doing grid-based neighbor search
#ifndef imax
#define imax( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef imin
#define imin( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

/**
* Check for CUDA errors; print and exit if there was a problem.
*/
void checkCUDAError(const char *msg, int line = -1) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    if (line >= 0) {
      fprintf(stderr, "Line %d: ", line);
    }
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}


/*****************
* Configuration *
*****************/

/*! Block size used for CUDA kernel launch. */
#define blockSize 128

// LOOK-1.2 Parameters for the boids algorithm.
// These worked well in our reference implementation.
#define rule1Distance 5.0f
#define rule2Distance 3.0f
#define rule3Distance 5.0f

#define rule1Scale 0.01f
#define rule2Scale 0.1f
#define rule3Scale 0.1f

#define maxSpeed 1.0f

/*! Size of the starting area in simulation space. */
#define scene_scale 100.0f

/***********************************************
* Kernel state (pointers are device pointers) *
***********************************************/

int numObjects;
dim3 threadsPerBlock(blockSize);

// LOOK-1.2 - These buffers are here to hold all your boid information.
// These get allocated for you in Boids::initSimulation.
// Consider why you would need two velocity buffers in a simulation where each
// boid cares about its neighbors' velocities.
// These are called ping-pong buffers.
glm::vec3 *dev_pos;
glm::vec3 *dev_vel1;
glm::vec3 *dev_vel2;

// LOOK-2.1 - these are NOT allocated for you. You'll have to set up the thrust
// pointers on your own too.

// For efficient sorting and the uniform grid. These should always be parallel.
int *dev_particleArrayIndices; // What tid in dev_pos and dev_velX represents this particle?
int *dev_particleGridIndices; // What grid cell is this particle in?
// needed for use with thrust
thrust::device_ptr<int> dev_thrust_particleArrayIndices;
thrust::device_ptr<int> dev_thrust_particleGridIndices;

int *dev_gridCellStartIndices; // What part of dev_particleArrayIndices belongs
int *dev_gridCellEndIndices;   // to this cell?

// TODO-2.3 - consider what additional buffers you might need to reshuffle
// the position and velocity data to be coherent within cells.

// LOOK-2.1 - Grid parameters based on simulation parameters.
// These are automatically computed for you in Boids::initSimulation
int gridCellCount;
int gridSideCount;
float gridCellWidth;
float gridInverseCellWidth;
glm::vec3 gridMinimum;

/******************
* initSimulation *
******************/

__host__ __device__ unsigned int hash(unsigned int a) {
  a = (a + 0x7ed55d16) + (a << 12);
  a = (a ^ 0xc761c23c) ^ (a >> 19);
  a = (a + 0x165667b1) + (a << 5);
  a = (a + 0xd3a2646c) ^ (a << 9);
  a = (a + 0xfd7046c5) + (a << 3);
  a = (a ^ 0xb55a4f09) ^ (a >> 16);
  return a;
}

/**
* LOOK-1.2 - this is a typical helper function for a CUDA kernel.
* Function for generating a random vec3.
*/
__host__ __device__ glm::vec3 generateRandomVec3(float time, int tid) {
  thrust::default_random_engine rng(hash((int)(tid * time)));
  thrust::uniform_real_distribution<float> unitDistrib(-1, 1);

  return glm::vec3((float)unitDistrib(rng), (float)unitDistrib(rng), (float)unitDistrib(rng));
}

/**
* LOOK-1.2 - This is a basic CUDA kernel.
* CUDA kernel for generating boids with a specified mass randomly around the star.
*/
__global__ void kernGenerateRandomPosArray(int time, int N, glm::vec3 * arr, float scale) {
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (tid < N) {
    glm::vec3 rand = generateRandomVec3(time, tid);
    arr[tid].x = scale * rand.x;
    arr[tid].y = scale * rand.y;
    arr[tid].z = scale * rand.z;
  }
}

/**
* Initialize memory, update some globals
*/
void Boids::initSimulation(int N) {
  numObjects = N;
  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

  // LOOK-1.2 - This is basic CUDA memory management and error checking.
  // Don't forget to cudaFree in  Boids::endSimulation.
  cudaMalloc((void**)&dev_pos, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_pos failed!");

  cudaMalloc((void**)&dev_vel1, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel1 failed!");

  cudaMalloc((void**)&dev_vel2, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel2 failed!");

  // Initialize velocity to 0
  cudaMemset(dev_vel1, 0, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMemset dev_vel1 failed!");

  cudaMemset(dev_vel2, 0, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMemset dev_vel2 failed!");

  // LOOK-1.2 - This is a typical CUDA kernel invocation.
  kernGenerateRandomPosArray<<<fullBlocksPerGrid, blockSize>>>(1, numObjects,
    dev_pos, scene_scale);
  checkCUDAErrorWithLine("kernGenerateRandomPosArray failed!");

  // LOOK-2.1 computing grid params
  gridCellWidth = 2.0f * std::max(std::max(rule1Distance, rule2Distance), rule3Distance);
  int halfSideCount = (int)(scene_scale / gridCellWidth) + 1;
  gridSideCount = 2 * halfSideCount;

  gridCellCount = gridSideCount * gridSideCount * gridSideCount;
  gridInverseCellWidth = 1.0f / gridCellWidth;
  float halfGridWidth = gridCellWidth * halfSideCount;
  gridMinimum.x -= halfGridWidth;
  gridMinimum.y -= halfGridWidth;
  gridMinimum.z -= halfGridWidth;

  // TODO-2.1 TODO-2.3 - Allocate additional buffers here.
  cudaMalloc((void**)&dev_particleArrayIndices, N * sizeof(int));
  cudaMalloc((void**)&dev_particleGridIndices, N * sizeof(int));
  cudaMalloc((void**)&dev_gridCellStartIndices, N * sizeof(int));
  cudaMalloc((void**)&dev_gridCellEndIndices, N * sizeof(int));

  // init the thrust pointers to point at same memory as the regular dev_ stuff
  dev_thrust_particleArrayIndices = thrust::device_ptr<int>(dev_particleArrayIndices);
  dev_thrust_particleGridIndices = thrust::device_ptr<int>(dev_particleGridIndices);

  cudaDeviceSynchronize();
}


/******************
* copyBoidsToVBO *
******************/

/**
* Copy the boid positions into the VBO so that they can be drawn by OpenGL.
*/
__global__ void kernCopyPositionsToVBO(int N, glm::vec3 *pos, float *vbo, float s_scale) {
  int tid = threadIdx.x + (blockIdx.x * blockDim.x);

  float c_scale = -1.0f / s_scale;

  if (tid < N) {
    vbo[4 * tid + 0] = pos[tid].x * c_scale;
    vbo[4 * tid + 1] = pos[tid].y * c_scale;
    vbo[4 * tid + 2] = pos[tid].z * c_scale;
    vbo[4 * tid + 3] = 1.0f;
  }
}

__global__ void kernCopyVelocitiesToVBO(int N, glm::vec3 *vel, float *vbo, float s_scale) {
  int tid = threadIdx.x + (blockIdx.x * blockDim.x);

  if (tid < N) {
    vbo[4 * tid + 0] = vel[tid].x + 0.3f;
    vbo[4 * tid + 1] = vel[tid].y + 0.3f;
    vbo[4 * tid + 2] = vel[tid].z + 0.3f;
    vbo[4 * tid + 3] = 1.0f;
  }
}

/**
* Wrapper for call to the kernCopyboidsToVBO CUDA kernel.
*/
void Boids::copyBoidsToVBO(float *vbodptr_positions, float *vbodptr_velocities) {
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

  kernCopyPositionsToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_pos, vbodptr_positions, scene_scale);
  kernCopyVelocitiesToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_vel1, vbodptr_velocities, scene_scale);

  checkCUDAErrorWithLine("copyBoidsToVBO failed!");

  cudaDeviceSynchronize();
}


/******************
* stepSimulation *
******************/

/**
* LOOK-1.2 You can use this as a helper for kernUpdateVelocityBruteForce.
* __device__ code can be called from a __global__ context
* Compute the new velocity on the body with tid `iSelf` due to the `N` boids
* in the `pos` and `vel` arrays.
*/
__device__ glm::vec3 computeVelocityChange(int N, int iSelf, const glm::vec3 *pos, const glm::vec3 *vel) {

  glm::vec3 rule1;
  glm::vec3 rule2;
  glm::vec3 rule3;

  glm::vec3 perceivedCOM;
  glm::vec3 offset;
  glm::vec3 perceivedVel;
  int numNeighbors = 0;

  // for each not iSelf
  for (int i = 0; i < N; i++) {
    if (i == iSelf) continue;

    // Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
    if (distance(pos[i], pos[iSelf]) < rule1Distance) {
      perceivedCOM += pos[i];
      numNeighbors++;
    };

    // Rule 2: boids try to stay a distance d away from each other
    if (distance(pos[i], pos[iSelf]) < rule2Distance) {
      offset -= pos[i] - pos[iSelf];
    };

    // Rule 3: boids try to match the speed of surrounding boids
    if (distance(pos[i], pos[iSelf]) < rule3Distance) {
      perceivedVel += vel[i];
    };
  }

  // Process rule 1 (guard against divide-by-zero when there are no neighbors)
  if (numNeighbors > 0) {
    perceivedCOM /= numNeighbors;
    rule1 = (perceivedCOM - pos[iSelf]) * rule1Scale;
    perceivedVel /= numNeighbors;
    rule3 = perceivedVel * rule3Scale;
  }
  rule2 = offset * rule2Scale;

  // glm::vec3 final = vel[iSelf] + rule1 + rule2 + rule3;
  // std::cout << glm::to_string(final) << std::endl;
  return vel[iSelf] + rule1 + rule2 + rule3;
}

/**
* TODO-1.2 implement basic flocking
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdateVelocityBruteForce(int N, glm::vec3 *pos,
  glm::vec3 *vel1, glm::vec3 *vel2) {
  // Compute a new velocity based on pos and vel1
  // Clamp the speed
  // Record the new velocity into vel2. Question: why NOT vel1?

  int tid = threadIdx.x + blockDim.x * blockIdx.x;

  glm::vec3 velNew = computeVelocityChange(N, tid, pos, vel1);
  if (glm::length(velNew) > maxSpeed) {
    velNew = glm::normalize(velNew) * maxSpeed;
  };

  vel2[tid] = velNew;
}

/**
* LOOK-1.2 Since this is pretty trivial, we implemented it for you.
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdatePos(int N, float dt, glm::vec3 *pos, glm::vec3 *vel) {
  // Update position by velocity
  int tid = threadIdx.x + (blockIdx.x * blockDim.x);
  if (tid >= N) {
    return;
  }
  glm::vec3 thisPos = pos[tid];
  thisPos += vel[tid] * dt;

  // Wrap the boids around so we don't lose them
  thisPos.x = thisPos.x < -scene_scale ? scene_scale : thisPos.x;
  thisPos.y = thisPos.y < -scene_scale ? scene_scale : thisPos.y;
  thisPos.z = thisPos.z < -scene_scale ? scene_scale : thisPos.z;

  thisPos.x = thisPos.x > scene_scale ? -scene_scale : thisPos.x;
  thisPos.y = thisPos.y > scene_scale ? -scene_scale : thisPos.y;
  thisPos.z = thisPos.z > scene_scale ? -scene_scale : thisPos.z;

  pos[tid] = thisPos;
}

// LOOK-2.1 Consider this method of computing a 1D tid from a 3D grid tid.
// LOOK-2.3 Looking at this method, what would be the most memory efficient
//          order for iterating over neighboring grid cells?
//          for(x)
//            for(y)
//             for(z)? Or some other order?
__device__ int gridIndex3Dto1D(int x, int y, int z, int cellsPerAxis) {
  return x + y * cellsPerAxis + z * cellsPerAxis * cellsPerAxis;
}

__device__ void gridIndex1Dto3D(int cellID, int cellsPerAxis, int &x, int &y, int &z) {
  int total = cellID;
  z = floorf(total / (cellsPerAxis * cellsPerAxis));
  total = total - (z * cellsPerAxis * cellsPerAxis);
  y = floorf(total / cellsPerAxis);
  x = total - (y * cellsPerAxis);
}

// COMPUTE INDICES (which become dev_particleArrayIndices)
// and GRID INDICES (which become dev_particleGridIndices)
__global__ void kernComputeIndices(int N, int cellsPerAxis,
  glm::vec3 cellGridMin, float inverseCellWidth,
  glm::vec3 *pos, int *indices, int *gridIndices) {
    // TODO-2.1
    // - Label each boid with the tid of its grid cell.
    // - Set up a parallel array of integer indices as pointers to the actual
    //   boid data in pos and vel1/vel2
    int tid = threadIdx.x + blockDim.x * blockIdx.x; 

    if (tid >= N) return;

    // cnvert pos[tid] to grid cell coord
    glm::vec3 gridPos = pos[tid] - cellGridMin; // relative to cell grid 
    int cellX = (int)std::floor(gridPos.x * inverseCellWidth); // convert world -> cell coords
    int cellY = (int)std::floor(gridPos.y * inverseCellWidth);
    int cellZ = (int)std::floor(gridPos.z * inverseCellWidth);

    // ok we sorting these
    // indices = dev_particleArrayIndices = what tid in dev_pos and dev_vel connects to PARTICLE
    indices[tid] = tid; // this looks like no-op but its how we'll know for when stuff gets sorted
    // gridIndices = dev_particleGridIndices = what grid CELL this PARTICLE in ????
    gridIndices[tid] = gridIndex3Dto1D(cellX, cellY, cellZ, cellsPerAxis);
}

// LOOK-2.1 Consider how this could be useful for indicating that a cell
//          does not enclose any boids
__global__ void kernResetIntBuffer(int N, int *intBuffer, int value) {
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (tid < N) {
    intBuffer[tid] = value;
  }
}

__global__ void kernIdentifyCellStartEnd(int N, int *particleGridIndices,
  int *gridCellStartIndices, int *gridCellEndIndices) {
  // TODO-2.1
  // Identify the start point of each cell in the gridIndices array.
  // This is basically a parallel unrolling of a loop that goes
  // "this tid doesn't match the one before it, must be a new cell!"
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= N) return;

  int cell = particleGridIndices[tid]; // grid cell this boid is in

  if (tid == N - 1) {
    gridCellEndIndices[cell] = tid;
  }

  if (tid == 0) {
    // if first boid, must be first tid for that cell
    gridCellStartIndices[cell] = tid;
  } else {
    int prev = particleGridIndices[tid - 1];
    if (cell != prev) { // if last boid is in diff cells than this boid
      gridCellStartIndices[cell] = tid;
      gridCellEndIndices[prev] = tid - 1;
    }
  }
}

__global__ void kernUpdateVelNeighborSearchScattered(
  int N, int cellsPerAxis, glm::vec3 cellGridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  int *particleArrayIndices, int *particleGridIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
  // TODO-2.1 - Update a boid's velocity using the uniform grid to reduce
  // the number of boids that need to be checked.
  // - Identify the grid cell that this particle is in
  // - Identify which cells may contain neighbors. This isn't always 8.
  // - For each cell, read the start/end indices in the boid pointer array.
  // - Access each boid in the cell and compute velocity change from
  //   the boids rules, if this boid is within the neighborhood distance.
  // - Clamp the speed change before putting the new speed in vel2

  // particle
  int tid = threadIdx.x + blockDim.x * blockIdx.x;

  if (tid >= N) return;

  int boidID = particleArrayIndices[tid];
  int cellID = particleGridIndices[tid];

  // boid info
  glm::vec3 boidPos = pos[boidID];
  glm::vec3 boidVel = vel1[boidID];

  int x, y, z; // we retrieve the cell coords in world coords
  gridIndex1Dto3D(cellID, cellsPerAxis, x, y, z);

  glm::vec3 boidPosWithinCell = boidPos - (cellGridMin + glm::vec3(x, y, z) * cellWidth);

  int xOffset = (boidPosWithinCell.x < cellWidth * 0.5f) ? -1 : 1;
  int yOffset = (boidPosWithinCell.y < cellWidth * 0.5f) ? -1 : 1;
  int zOffset = (boidPosWithinCell.z < cellWidth * 0.5f) ? -1 : 1;

  glm::vec3 perceivedCOM;
  glm::vec3 offset;
  glm::vec3 perceivedVel;
  int numNeighbors = 0;

  // for this boid + all boids in its 8 cell territory
  // dx innermost bc better locality? cache friendly mem access
  // x is fastest varying term
  for (int dz = 0; dz < 2; dz++) {
    for (int dy = 0; dy < 2; dy++) {
      for (int dx = 0; dx < 2; dx++) {
        // when dz, dy, dx are all 0 we're looking at our original cellID
        // ow, we set the offsets accordingly to get neighbors
        int nx = x + (dx ? xOffset : 0);
        int ny = y + (dy ? yOffset : 0);
        int nz = z + (dz ? zOffset : 0);

        if (nx < 0 || nx >= cellsPerAxis || ny < 0 || ny >= cellsPerAxis
        || nz < 0 || nz >= cellsPerAxis) continue;

        int neighborCell = gridIndex3Dto1D(nx, ny, nz, cellsPerAxis);
        int start = gridCellStartIndices[neighborCell];
        int end = gridCellEndIndices[neighborCell];

        if (start == -1) continue;

        // for all indices in gridArray marked by start/endarray
        for (int i = start; i < end + 1; i++) {
          int thisBoidID = particleArrayIndices[i];
          if (thisBoidID == boidID) continue;

          glm::vec3 thisBoidPos = pos[thisBoidID];
          float dist = distance(thisBoidPos, boidPos);

          // RULE 1 (com)
          if (dist < rule1Distance) {
            perceivedCOM += thisBoidPos;
            numNeighbors++;
          }
          // RULE 2 (stay distance away)
          if (dist < rule2Distance) {
            offset -= (thisBoidPos - boidPos);
          }
          // RULE 3 (match velocity of nearby bods)
          if (dist < rule3Distance) {
            perceivedVel += vel1[thisBoidID];
          }
        }

      }
    }
  }

  glm::vec3 rule1, rule2, rule3;
  
  if (numNeighbors > 0) {
    // process stuff
    perceivedCOM /= numNeighbors;
    rule1 = (perceivedCOM - boidPos) * rule1Scale;
    perceivedVel /= numNeighbors;
    rule3 = perceivedVel * rule3Scale;
  };
  rule2 = offset * rule2Scale;

  glm::vec3 velNew = vel1[boidID] + rule1 + rule2 + rule3;
  // clamp
  if (glm::length(velNew) > maxSpeed) {
    velNew = glm::normalize(velNew) * maxSpeed;
  }

  vel2[boidID] = velNew;
}

__global__ void kernUpdateVelNeighborSearchCoherent(
  int N, int cellsPerAxis, glm::vec3 cellGridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
  // TODO-2.3 - This should be very similar to kernUpdateVelNeighborSearchScattered,
  // except with one less level of indirection.
  // This should expect gridCellStartIndices and gridCellEndIndices to refer
  // directly to pos and vel1.
  // - Identify the grid cell that this particle is in
  // - Identify which cells may contain neighbors. This isn't always 8.
  // - For each cell, read the start/end indices in the boid pointer array.
  //   DIFFERENCE: For best results, consider what order the cells should be
  //   checked in to maximize the memory benefits of reordering the boids data.
  // - Access each boid in the cell and compute velocity change from
  //   the boids rules, if this boid is within the neighborhood distance.
  // - Clamp the speed change before putting the new speed in vel2


}

/**
* Step the entire N-body simulation by `dt` seconds.
*/
void Boids::stepSimulationNaive(float dt) {
  // TODO-1.2 - use the kernels you wrote to step the simulation forward in time.
  // blocksPerGrid, threadsperblock
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
  kernUpdateVelocityBruteForce<<<fullBlocksPerGrid, blockSize>>>(numObjects, dev_pos, dev_vel1, dev_vel2);
  checkCUDAErrorWithLine("kernUpdateVelocityBruteForce fail");

  kernUpdatePos<<<fullBlocksPerGrid, blockSize>>>(numObjects, dt, dev_pos, dev_vel2);
  checkCUDAErrorWithLine("kernUpdatePos fail");

  // TODO-1.2 ping-pong the velocity buffers
  std::swap(dev_vel1, dev_vel2);
}

void Boids::stepSimulationScatteredGrid(float dt) {
  // TODO-2.1
  // Uniform Grid Neighbor search using Thrust sort.
  // In Parallel:
  // - label each particle with its array tid as well as its grid tid.
  //   Use 2x width grids.
  // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
  //   are welcome to do a performance comparison.
  // - Naively unroll the loop for finding the start and end indices of each
  //   cell's data pointers in the array of boid indices
  // - Perform velocity updates using neighbor search
  // - Update positions
  // - Ping-pong buffers as needed

  dim3 fullBlocksPerGrid((numObjects + blockSize - 1)/ blockSize);
  dim3 blocksPerGridCells((gridCellCount + blockSize -1) / blockSize);

  // Compute grid indices for each boid based on its position
  kernComputeIndices<<<fullBlocksPerGrid, blockSize>>>(numObjects, gridSideCount, gridMinimum,
  gridInverseCellWidth, dev_pos, dev_particleArrayIndices, dev_particleGridIndices);
  checkCUDAErrorWithLine("kernComputeIndices fail");

  // Sort boids by grid (key)
  thrust::sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + numObjects, dev_thrust_particleArrayIndices);

  // Reset start/end arrays each frame
  kernResetIntBuffer<<<blocksPerGridCells, blockSize>>>(gridCellCount, dev_gridCellStartIndices, -1);
  kernResetIntBuffer<<<blocksPerGridCells, blockSize>>>(gridCellCount, dev_gridCellEndIndices, -1);
  // Cell start & end (using particlegridindicies, which is )
  kernIdentifyCellStartEnd<<<fullBlocksPerGrid, blockSize>>>(numObjects, dev_particleGridIndices, dev_gridCellStartIndices, dev_gridCellEndIndices);

  // actually compute vel (new velocity written to dev2)
  kernUpdateVelNeighborSearchScattered<<<fullBlocksPerGrid, blockSize>>>(numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, gridCellWidth, 
  dev_gridCellStartIndices, dev_gridCellEndIndices, dev_particleArrayIndices, dev_particleGridIndices, dev_pos, dev_vel1, dev_vel2);
  checkCUDAErrorWithLine("kernUpdateVelNeighborSearchScattered fail");

  kernUpdatePos<<<fullBlocksPerGrid, blockSize>>>(numObjects, dt, dev_pos, dev_vel2);
  checkCUDAErrorWithLine("kernUpdatePos fail");

  std::swap(dev_vel1, dev_vel2);
}

void Boids::stepSimulationCoherentGrid(float dt) {
  // TODO-2.3 - start by copying Boids::stepSimulationNaiveGrid
  // Uniform Grid Neighbor search using Thrust sort on cell-coherent data.
  // In Parallel:
  // - Label each particle with its array tid as well as its grid tid.
  //   Use 2x width grids
  // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
  //   are welcome to do a performance comparison.
  // - Naively unroll the loop for finding the start and end indices of each
  //   cell's data pointers in the array of boid indices
  // - BIG DIFFERENCE: use the rearranged array tid buffer to reshuffle all
  //   the particle data in the simulation array.
  //   CONSIDER WHAT ADDITIONAL BUFFERS YOU NEED
  // - Perform velocity updates using neighbor search
  // - Update positions
  // - Ping-pong buffers as needed. THIS MAY BE DIFFERENT FROM BEFORE.
}

void Boids::endSimulation() {
  cudaFree(dev_vel1);
  cudaFree(dev_vel2);
  cudaFree(dev_pos);

  // TODO-2.1 TODO-2.3 - Free any additional buffers here.
  cudaFree(dev_particleArrayIndices);
  cudaFree(dev_particleGridIndices);
  cudaFree(dev_gridCellStartIndices);
  cudaFree(dev_gridCellEndIndices);
}

void Boids::unitTest() {
  // LOOK-1.2 Feel free to write additional tests here.

  // test unstable sort
  int *dev_intKeys;
  int *dev_intValues;
  int N = 10;

  std::unique_ptr<int[]>intKeys{ new int[N] };
  std::unique_ptr<int[]>intValues{ new int[N] };

  intKeys[0] = 0; intValues[0] = 0;
  intKeys[1] = 1; intValues[1] = 1;
  intKeys[2] = 0; intValues[2] = 2;
  intKeys[3] = 3; intValues[3] = 3;
  intKeys[4] = 0; intValues[4] = 4;
  intKeys[5] = 2; intValues[5] = 5;
  intKeys[6] = 2; intValues[6] = 6;
  intKeys[7] = 0; intValues[7] = 7;
  intKeys[8] = 5; intValues[8] = 8;
  intKeys[9] = 6; intValues[9] = 9;

  cudaMalloc((void**)&dev_intKeys, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_intKeys failed!");

  cudaMalloc((void**)&dev_intValues, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_intValues failed!");

  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

  std::cout << "before unstable sort: " << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << "  key: " << intKeys[i];
    std::cout << " value: " << intValues[i] << std::endl;
  }

  // How to copy data to the GPU
  cudaMemcpy(dev_intKeys, intKeys.get(), sizeof(int) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_intValues, intValues.get(), sizeof(int) * N, cudaMemcpyHostToDevice);

  // Wrap device vectors in thrust iterators for use with thrust.
  thrust::device_ptr<int> dev_thrust_keys(dev_intKeys);
  thrust::device_ptr<int> dev_thrust_values(dev_intValues);
  // LOOK-2.1 Example for using thrust::sort_by_key
  thrust::sort_by_key(dev_thrust_keys, dev_thrust_keys + N, dev_thrust_values);

  // How to copy data back to the CPU side from the GPU
  cudaMemcpy(intKeys.get(), dev_intKeys, sizeof(int) * N, cudaMemcpyDeviceToHost);
  cudaMemcpy(intValues.get(), dev_intValues, sizeof(int) * N, cudaMemcpyDeviceToHost);
  checkCUDAErrorWithLine("memcpy back failed!");

  std::cout << "after unstable sort: " << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << "  key: " << intKeys[i];
    std::cout << " value: " << intValues[i] << std::endl;
  }

  // cleanup
  cudaFree(dev_intKeys);
  cudaFree(dev_intValues);
  checkCUDAErrorWithLine("cudaFree failed!");
  return;
}
