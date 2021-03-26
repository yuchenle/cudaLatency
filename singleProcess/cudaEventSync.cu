#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "../cudaErr.h"

__global__ void dummy_kernel()
{
}

int main()
{
  struct timespec stamp, previous_stamp;
  double wtime;
  cudaStream_t st1, st2;
  gpuErrchk (cudaStreamCreate (&st1));
  gpuErrchk (cudaStreamCreate (&st2));

  cudaEvent_t et; 
  gpuErrchk (cudaEventCreate (&et));

  for (int i = 0; i < 1000000; i++)
  {
    clock_gettime (CLOCK_MONOTONIC, &previous_stamp);

    dummy_kernel <<<1, 1, 0, st1>>>();
    gpuErrchk (cudaEventRecord (et, st1));
    gpuErrchk (cudaEventSynchronize(et));

    dummy_kernel <<<1, 1, 0, st2>>>();
    gpuErrchk (cudaEventRecord (et, st2));
    gpuErrchk (cudaEventSynchronize(et));

    clock_gettime (CLOCK_MONOTONIC, &stamp);
    wtime = (stamp.tv_sec - previous_stamp.tv_sec) * 1000000000 + (stamp.tv_nsec - previous_stamp.tv_nsec);
    printf ("%.4f \n", wtime);
  }
  
}
