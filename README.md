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
From what I've seen in the docs, `cudaGetDeviceCount` can never return a value below 0. I wanted to ask if (gpuDevice > device_count) was meant to be (gpuDevice >= device_count). I understand that we set gpuDevice to 0 because we want to use the first device, but this if check looks like it shouldn't do anything unless we change the equality to >=. 
I apologize if this was already pointed out by someone else, or if the code is correct!

### (TODO: Your README)

Include screenshots, analysis, etc. (Remember, this is public, so don't put
anything here that you don't want to share with the world.)
