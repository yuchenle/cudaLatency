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
  clock_gettime (CLOCK_MONOTONIC, &stamp);
  double wtime;
  cudaStream_t st1;
  gpuErrchk (cudaStreamCreate (&st1));
  
  for (int i = 0; i < 1000000; i++)
  {
    dummy_kernel <<<1, 1, 0, st1>>>();
    dummy_kernel <<<1, 1, 0, st1>>>();
    gpuErrchk (cudaStreamSynchronize(st1));
    memcpy (&previous_stamp, &stamp, sizeof (struct timespec));
    clock_gettime (CLOCK_MONOTONIC, &stamp);
    wtime = (stamp.tv_sec - previous_stamp.tv_sec) * 1000000 + (stamp.tv_nsec - previous_stamp.tv_nsec) / 1000;
    printf ("%.4f \n", wtime);
  }
  
}
