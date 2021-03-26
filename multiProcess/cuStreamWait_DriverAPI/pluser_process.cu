extern "C"
{
  #include <cuda.h>
  #include "const.h"
  __global__ void pluser_process (GPU_DATA_TYPE *data)
  {
    int threadId = (blockIdx.x * blockDim.x) + threadIdx.x;
    for (;threadId < GPU_SIZE; threadId += blockDim.x * gridDim.x)
      data[threadId] += 1;
  }
}
