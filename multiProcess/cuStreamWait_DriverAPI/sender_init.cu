extern "C"
{
  #include <cuda.h>
  #include "const.h"
  __global__ void  sender_init (GPU_DATA_TYPE *data)
  {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    while (i < GPU_SIZE)
    {
      data[i]=1.0;
      i += blockDim.x * gridDim.x;
    }
    i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i == 0)
      printf ("init done, first element is %.4f\n", data[0]);
  }
}
