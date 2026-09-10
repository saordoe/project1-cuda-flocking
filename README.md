**University of Pennsylvania, CIS 5650: GPU Programming and Architecture,
Project 1 - Flocking**

* Jimin Choi
  * [LinkedIn](https://linkedin.com/in/jiminchoi4) | [Personal Website](https://jiminchoi.com/) | [X](https://x.com/saordoe)
* Tested on: Windows 11 Education, i9-12900F @ 2.40 GHz 64GB, NVIDIA GeForce RTX 3090 24GB (Lab Computer)
* GPU Compute Compatibility: 8.6

### Assignment Feedback
I noticed that in `main.cpp` of the base code, we have the following code in `init()`:
```cpp
  cudaDeviceProp deviceProp;
  int gpuDevice = 0;
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (gpuDevice > device_count) {
    std::cout
    << "Error: GPU device number is greater than the number of devices!"
    << " Perhaps a CUDA-capable GPU is not installed?"
    << std::endl;
    return false;
  }
```
From what I've seen in the docs, `cudaGetDeviceCount` can never return a value below 0. I wanted to ask if `gpuDevice > device_count` was meant to be `gpuDevice >= device_count`. I understand that we set `gpuDevice` to 0 because we want to use the first device, but this `if` check seemed like dead code, unless the equality was meant to be `>=`. 
I apologize if this was already pointed out by someone else, or if the code is correct!

# CUDA Boids

## Visualizations

### 50K Boids (DT = 0.08)
<p align="center">
  <img src="images/cudaBoids1small.gif" alt="boids_gif">
</p>

### 50K Boids (DT = 0.1)
<p align="center">
  <img src="images/cudaBoids5small.gif" alt="boids_gif">
</p>

### 20K Boids (DT = 0.1)

<p align="center">
  <img src="images/cudaBoids6.gif" alt="boids_gif">
</p>

An observation worth noting is that with a higher boid count, boids' trajectories seem to converge at a much more rapid rate. In the 20K simulations, boids' flocking patterns vary and stay varied in comparison to the 50K simulations. 

## Performance Analysis

We explored three separate approaches to recreating [Conard Parker's](http://www.vergenet.net/~conrad/boids/pseudocode.html) boids simulation in CUDA. The first was a naive approach where every boid checks every other boid in the simulation to compute its velocity, resulting in O(N^2) work per timestep.

We then explored two optimizations: a scattered uniform grid, which buckets boids into grid cells and only checks neighboring cells, and a coherent uniform grid, which additionally reshuffles boid position and velocity data to be contiguous in memory per cell, improving memory access patterns during neighbor search.

Below are visual charts representing the performance (frames per second) of each implementation as boid count is varied. All versions of the simulation were tested at each 15K boids, 20K boids, 50K boids, and 100K boids.

<p align="center">
  <img src="images/withoutvis.png" alt="withoutvis">
</p>

<p align="center">
  <img src="images/withvis.png" alt="withvis">
</p>

As expected, performance using the naive approach declines sharply over increased boid count, with as much as a 91.8 percent decrease in FPS between the 20K and 100K simulation results. Performance decline in the coherent and scattered approaches is shown to be much less substantial in comparison, with the coherent model most always performing slightly better than the scattered model with any boid count. Contrast to expectation, for both the coherent and scattered approaches demonstrated a sudden decline in performance when run with 50,000 boids. In fact, both simulations performed better at a 100,000 boid count than at 50,000. 

We ran an analysis this time varying the block sizes, with a boid count fixed at 100K boids.

<p align="center">
  <img src="images/blocksize.png" alt="blocksize">
</p>

For all three approaches, there seemed to be common ground in that performance peaked at the lower half of graph, between 64 to 512 threads per block. We can deduce from this performance graph that the work items throughout our program have higher variability in terms of the amount of time it takes to fulfill each work item.