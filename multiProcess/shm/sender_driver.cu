#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <errno.h>

#include "../cudaErr.h"
#include "const.h"
#include "Sender.hpp"

// global variables to be initialized
int shmid;
TYPE *ptr;

void init_value (TYPE *ptr, TYPE value)
{
  TYPE *tmp_ptr = ptr;
  for (int i = 0; i < SIZE; i++)
  {
    *tmp_ptr = WAIT_VALUE;
    tmp_ptr ++;
  }
  // * (ptr + (SIZE - 1)) = WAIT_VALUE;
}

void init ()
{
  // FILE to key
  key_t key = ftok (FILENAME, FILEID);
  if (key == -1) 
  {
    printf ("ftok failed, errno = %s\n", strerror(errno));
    exit (-1);
  }

  // getting SHM id
  printf (" size of shm is %zu\n", sizeof(TYPE)*SIZE+sizeof(cudaIpcMemHandle_t));
  shmid = shmget (key, sizeof (TYPE) * SIZE + sizeof (cudaIpcMemHandle_t), 0666|IPC_CREAT);
  if (shmid == -1) 
  {
    printf ("shmget failed, errno = %s\n", strerror(errno));
    exit (-1);
  }

  // attach the SHM to this process
  ptr = (TYPE *) shmat (shmid, (void *)0, 0); 

  init_value (ptr, 0);
}

__global__ void init_kernel (GPU_DATA_TYPE *d_data)
{
  int threadId = (blockIdx.x * blockDim.x) + threadIdx.x;
  for (;threadId < GPU_SIZE; threadId += blockDim.x * gridDim.x)
  {
    d_data[threadId] = 1;
  }
}

int main()
{

  init ();
  printf ("ptr is %p\n", (void *)ptr);


  GPU_DATA_TYPE *d_a;
  gpuErrchk (cudaMalloc (&d_a, sizeof(int) * GPU_SIZE));

  init_kernel<<<128,1024>>> (d_a);

  Sender *sender = new Sender(shmid, ptr, d_a);

  cudaIpcMemHandle_t *handle = (cudaIpcMemHandle_t *)malloc (sizeof (cudaIpcMemHandle_t));
  gpuErrchk (cudaIpcGetMemHandle (handle, d_a));
  sender->set_GPUIPC_handle (handle);

  // copy the handle in the SHM
  unsigned char *p = (unsigned char *) ptr;
  memcpy (p+SIZE, handle, sizeof(cudaIpcMemHandle_t));
  printf ("sender: shm_ptr is pointing to %p, and p+1 at %p\n", (void *)ptr, (void *)(p+1));

  // main loop
  // while (true)
  for (int i = 0; i < 1000000; i++)
  {
    sender->update ();
    sender->wait ();
    sender->process ();
    sender->notify ();
    // usleep (20);
  }
}
