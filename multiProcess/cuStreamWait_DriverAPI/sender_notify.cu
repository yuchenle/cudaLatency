extern "C" 
{
  #include <cuda.h>
  #include "const.h"
  __global__ void sender_notify (GPU_DATA_TYPE *data)
  {
    // printf ("sender::notify\n");
    data[0] = 1.0;
    __threadfence();
  }
}
