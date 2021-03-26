extern "C"
{
  #include <cuda.h>
  #include "const.h"
  __global__ void sender_wait (GPU_DATA_TYPE *data)
  {
    printf ("sender::wait, first element is %.4f\n", data[0]);
    volatile GPU_DATA_TYPE value = (volatile GPU_DATA_TYPE )*data;
    // TODO ,change to cuStreamWaitValue
    while (value == 1) 
    { 
      value = data[0];
      __threadfence();
    }
  }
}
