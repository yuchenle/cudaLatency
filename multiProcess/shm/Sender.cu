#include <stdio.h>
#include <unistd.h>
#include <string.h>

#include "Sender.hpp"
#include "../cudaErr.h"

__global__ void kernel(GPU_DATA_TYPE *);
__global__ void empty_kernel();

__global__ void empty_kernel()
{
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
  while (*Sender::ptr != WAIT_VALUE)
  {
      asm volatile("" ::: "memory");
  }
}

void Sender::process()
{
  // printf ("sender: starting process()\n");
  kernel_wrapper();
  // empty_kernel<<<1,1>>>(); // for latency measurement purpose
}

void Sender::notify()
{
  // printf ("sender: starting notify()\n");
  *ptr = !WAIT_VALUE;
}

void Sender::kernel_wrapper()
{
  // kernel <<<128, 1024>>> (d_data);
  empty_kernel<<<1,1024>>>();
  gpuErrchk (cudaDeviceSynchronize ());
}

__global__ void kernel(GPU_DATA_TYPE *d_ptr)
{
  int threadId = (blockIdx.x * blockDim.x) + threadIdx.x;
  for (;threadId < GPU_SIZE; threadId += blockDim.x * gridDim.x)
  {
    d_ptr[threadId] -= 1;
  }
}

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
