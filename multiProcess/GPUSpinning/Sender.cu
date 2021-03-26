#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <cuda.h>

#include "Sender.hpp"
#include "../cudaErr.h"

namespace SENDER
{
  __global__ void emptyKernel() {}
  __global__ void kernel(GPU_DATA_TYPE *data)
  {
    int threadId = (blockIdx.x * blockDim.x) + threadIdx.x;
    for (;threadId < GPU_SIZE-1; threadId += blockDim.x * gridDim.x)
    {
      data[threadId] -= 1;
    }
  }
  
  // This kernel should be launched by 1 thread 
  __global__ void notify_kernel (GPU_DATA_TYPE *data)
  {
    data[GPU_SIZE-1] = 1;
  }
  
  // This kernel should be launched by 1 thread 
  __global__ void wait_kernel (GPU_DATA_TYPE *data)
  {
    while (data[GPU_SIZE-1] == 1)
    {
      __threadfence();
    }
  }
}

Sender::Sender ()
{
  Sender::shmid = -1;
  Sender::ptr = 0;
  Sender::mem_handle = (cudaIpcMemHandle_t *)malloc(sizeof (cudaIpcMemHandle_t));
  Sender::d_data = 0;
}

Sender::Sender (int id, TYPE *shm_ptr, GPU_DATA_TYPE *device_data)
{
  Sender::shmid = id;
  Sender::ptr = shm_ptr;
  Sender::mem_handle = (cudaIpcMemHandle_t *)malloc(sizeof (cudaIpcMemHandle_t));
  Sender::d_data = device_data;
}

Sender::~Sender ()
{
}

void Sender::update()
{
  // printf ("sender: starting update()\n");
}

void Sender::wait()
{
  // printf ("sender: starting wait()\n");
  SENDER::wait_kernel<<<1, 1>>> (d_data);
  // gpuErrchk (cudaDeviceSynchronize ());
}

void Sender::process()
{
  // printf ("sender: starting process()\n");
  // SENDER::kernel <<<4, 1024>>> (d_data);
  // gpuErrchk (cudaDeviceSynchronize ());
  SENDER::emptyKernel<<<1,1024>>>();
}

void Sender::notify()
{
  // printf ("sender: starting notify()\n");
  SENDER::notify_kernel<<<1,1>>> (d_data);
  // gpuErrchk (cudaDeviceSynchronize ());
}

/*
__global__ void wait_kernel (GPU_DATA_TYPE *data)
{
  int threadId = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (threadId == 0) 
  {
    printf ("sender::wait_kernel \n");
    while (data[GPU_SIZE-1] == 1)
    {}
  }
}

__global__ void notify_kernel(GPU_DATA_TYPE *data)
{
  int threadId = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (threadId == 0)
  {
    printf ("sender::notify kernel\n");
    data[GPU_SIZE-1] = 1;
  }
}

__global__ void kernel(GPU_DATA_TYPE *d_ptr)
{
  int threadId = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (threadId == 0)
    printf ("sender: executing kernel : first element is %.2f\n", d_ptr[0]);
  for (;threadId < GPU_SIZE-1; threadId += blockDim.x * gridDim.x)
  {
    d_ptr[threadId] += 1;
  }
}
*/


// Getters
int Sender::get_SHM_id()
{
  return Sender::shmid;
}

TYPE *Sender::get_SHM_ptr()
{
  return Sender::ptr;
}

cudaIpcMemHandle_t *Sender::get_GPUIPC_handle()
{
  return Sender::mem_handle;
}

GPU_DATA_TYPE *Sender::get_d_data()
{
  return Sender::d_data;
}

// Setters
void Sender::set_SHM_id (int id)
{
  Sender::shmid = id;
}

void Sender::set_SHM_ptr (TYPE *ptr)
{
  Sender::ptr = ptr;
}

void Sender::set_GPUIPC_handle (cudaIpcMemHandle_t *handle)
{
  memcpy (mem_handle, handle, sizeof (cudaIpcMemHandle_t));
}

void Sender::set_d_data (GPU_DATA_TYPE *device_data)
{
  Sender::d_data = device_data;
}
