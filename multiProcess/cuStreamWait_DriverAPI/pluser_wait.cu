extern "C"
{
  #include <cuda.h>
  #include "const.h"
  __global__ void pluser_wait (GPU_DATA_TYPE *data)
  {
    // printf ("pluser::wait, first element is %.4f\n", data[0]);
    while (data[0] == 0)
    {
      __threadfence();
    }
  }
}
