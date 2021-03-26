#include <stdio.h>
#include <unistd.h>
#include <string.h>

#include "Pluser.hpp"
#include "../cudaErr.h"

__global__ void kernel(GPU_DATA_TYPE *);
__global__ void empty_kernel()
{
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
  // printf ("pluser: starting update()\n");
}

void Pluser::wait()
{
  // printf ("pluser: starting wait()\n");
  while (*Pluser::ptr == WAIT_VALUE)
  {
      // usleep (1000);
      asm volatile("" ::: "memory");
  }

  if (d_data == 0)
  {
    printf ("pluser: in wait (), first time so setting mem_handle by memcpy\n");
    unsigned char *p = (unsigned char *)ptr;
    memcpy (mem_handle, p+SIZE, sizeof(cudaIpcMemHandle_t));

    printf ("pluser: in wait(), first time so setting d_data\n");
    gpuErrchk (cudaIpcOpenMemHandle ((void **)&d_data, *mem_handle, cudaIpcMemLazyEnablePeerAccess));
  }
}

void Pluser::process()
{
  // printf ("pluser: starting process()\n");
  kernel_wrapper();
}

void Pluser::notify()
{
  // printf ("pluser: starting notify()\n");
  *(ptr) = WAIT_VALUE;
  *(ptr+1) = !WAIT_VALUE;
}

void Pluser::kernel_wrapper()
{
  // kernel <<<128, 1024>>> (d_data);
  empty_kernel<<<1,1>>>();
  // gpuErrchk (cudaDeviceSynchronize ());
}

__global__ void kernel(GPU_DATA_TYPE *d_ptr)
{
  int threadId = (blockIdx.x * blockDim.x) + threadIdx.x;
  for (;threadId < GPU_SIZE; threadId += blockDim.x * gridDim.x)
    d_ptr[threadId] += 1;
}

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

void Pluser::set_GPUIPC_handle (cudaIpcMemHandle_t *handle)
{
  memcpy (mem_handle, handle, sizeof (cudaIpcMemHandle_t));
}

void Pluser::set_d_data (GPU_DATA_TYPE *device_data)
{
  Pluser::d_data = device_data;
}
