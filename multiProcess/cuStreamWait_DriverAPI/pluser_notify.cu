extern "C" 
{
  #include <cuda.h>
  #include "const.h"
  __global__ void pluser_notify (GPU_DATA_TYPE *data)
  {
    // printf ("pluser::notify\n");
    data[0] = 0.0;
  }
}
