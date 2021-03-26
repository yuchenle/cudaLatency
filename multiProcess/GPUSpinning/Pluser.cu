#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <cuda.h>

#include "Pluser.hpp"
#include "../cudaErr.h"


namespace PLUSER
{
  __global__ void emptyKernel() {}
  
  __global__ void kernel(GPU_DATA_TYPE *data)
  {
    int threadId = (blockIdx.x * blockDim.x) + threadIdx.x;
    for (;threadId < GPU_SIZE-1; threadId += blockDim.x * gridDim.x)
      data[threadId] += 1;
  }
  
  // this kernel should be launched by 1 thread
  __global__ void wait_kernel (GPU_DATA_TYPE *data)
  {
    while (data[GPU_SIZE - 1] == 0)
    {
      __threadfence();
    }
  }
  
  // this kernel should be launched by 1 thread
  __global__ void notify_kernel (GPU_DATA_TYPE *data)
  {
    data[GPU_SIZE-1] = 0;
  }
}

Pluser::Pluser ()
{
  Pluser::shmid = -1;
  Pluser::ptr = 0;
  Pluser::mem_handle = (cudaIpcMemHandle_t *)malloc(sizeof (cudaIpcMemHandle_t));
  Pluser::d_data = 0;
}

Pluser::Pluser (int id, TYPE *shm_ptr)
{
  Pluser::shmid = id;
  Pluser::ptr = shm_ptr;
  Pluser::mem_handle = (cudaIpcMemHandle_t *)malloc(sizeof (cudaIpcMemHandle_t));
  Pluser::d_data = 0;
}

Pluser::~Pluser ()
{
}

void Pluser::update()
{
//  printf ("pluser: starting update()\n");
}

void Pluser::wait()
{
  // printf ("pluser: starting wait()\n");
  PLUSER::wait_kernel<<<1, 1024>>>(d_data);
  // gpuErrchk (cudaDeviceSynchronize ());
}


void Pluser::process()
{
  // printf ("pluser: starting process()\n");
  // PLUSER::kernel <<<128, 1024, 0, ptr->RT_stream>>> (d_data);
  // PLUSER::kernel <<<4, 1024>>> (d_data);
  // gpuErrchk (cudaDeviceSynchronize ());
  PLUSER::emptyKernel<<<1,1024>>>();
}

void Pluser::notify()
{
  // printf ("pluser: starting notify()\n");
  // PLUSER::notify_kernel<<<1, 1, 0, ptr->RT_stream>>> (d_data);
  PLUSER::notify_kernel<<<1, 1024>>> (d_data);
  // gpuErrchk (cudaDeviceSynchronize ());
}

/*
__global__ void notify_kernel (GPU_DATA_TYPE *data)
{
  int threadId = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (threadId == 0)
  {
    printf ("pluser: notify kernel\n");
    data[GPU_SIZE-1] = 0;
  }
}

__global__ void wait_kernel (GPU_DATA_TYPE *data)
{
  int threadId = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (threadId == 0)
  {
    printf ("pluser: wait kernel\n");
    while (data[GPU_SIZE - 1] == 0)
    {}
  }
}

__global__ void kernel(GPU_DATA_TYPE *d_ptr)
{
  int threadId = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (threadId == 0)
    printf ("pluser: executing kernel, first element is %.2f\n", d_ptr[0]);
  for (;threadId < GPU_SIZE-1; threadId += blockDim.x * gridDim.x)
    d_ptr[threadId] += 1;
}
*/

// Getters
int Pluser::get_SHM_id()
{
  return Pluser::shmid;
}

TYPE *Pluser::get_SHM_ptr()
{
  return Pluser::ptr;
}

cudaIpcMemHandle_t *Pluser::get_GPUIPC_handle()
{
  return Pluser::mem_handle;
}

GPU_DATA_TYPE *Pluser::get_d_data()
{
  return Pluser::d_data;
}

// Setters
void Pluser::set_SHM_id (int id)
{
  Pluser::shmid = id;
}

void Pluser::set_SHM_ptr (TYPE *ptr)
{
  Pluser::ptr = ptr;
}

void Pluser::set_GPUIPC_handle (cudaIpcMemHandle_t *shm_handle)
{
  if (shm_handle == NULL) printf ("handle is null\n");
  if (Pluser::mem_handle == NULL) printf ("pluser::mem_handle is null\n");

  memcpy (mem_handle, shm_handle, sizeof (cudaIpcMemHandle_t));
}

void Pluser::set_d_data (GPU_DATA_TYPE *device_data)
{
  Pluser::d_data = device_data;
}
