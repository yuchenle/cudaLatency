#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <errno.h>
#include <signal.h>
#include <cuda_profiler_api.h>
#include <cuda.h>

#include "../cudaErr.h"
#include "const.h"
#include "Sender.hpp"

// global variables to be initialized
int shmid;
TYPE *ptr;
cudaStream_t RT_stream_g;
cudaStream_t callBack_stream_g;
cudaEvent_t event_g;

void init_shm (TYPE *ptr)
{
  for (int i = 0; i < SIZE; i++)
    ptr->ready[i] = WAIT_VALUE;

  gpuErrchk (cudaStreamCreate (&RT_stream_g));
  gpuErrchk (cudaStreamCreate (&callBack_stream_g));

  memcpy (&(ptr->RT_stream), &RT_stream_g, sizeof(cudaStream_t)); 
  memcpy (&(ptr->callBackStream), &callBack_stream_g, sizeof(cudaStream_t)); 


  // gpuErrchk (cudaStreamCreate (&(ptr->RT_stream)));
  // gpuErrchk (cudaStreamCreate (&(ptr->callBackStream)));
  // ptr->RT_stream = RT_stream_g;
  // ptr->callBackStream = &callBack_stream_g;
  // ptr->memHandle = (cudaIpcMemHandle_t *)malloc (sizeof (cudaIpcMemHandle_t));
  // ptr->eventHandle = (cudaIpcEventHandle_t *)malloc (sizeof (cudaIpcEventHandle_t));
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
  printf (" size of shm is %zu\n", sizeof(TYPE));
  shmid = shmget (key, sizeof (TYPE), 0666|IPC_CREAT);
  if (shmid == -1) 
  {
    printf ("shmget failed, errno = %s\n", strerror(errno));
    exit (-1);
  }

  // attach the SHM to this process
  ptr = (TYPE *) shmat (shmid, (void *)0, 0); 

  init_shm (ptr);
}

__global__ void init_kernel (GPU_DATA_TYPE *d_data)
{
  int threadId = (blockIdx.x * blockDim.x) + threadIdx.x;

  while (threadId < GPU_SIZE)
  {
    d_data[threadId] = 1;
    threadId += blockDim.x * gridDim.x;
  }
}

void sigInt_handler (int sig)
{
  if (sig == SIGINT)
  {
    printf ("sender received SIGINT, calling cudaProfilerStop before exiting\n");
    gpuErrchk (cudaProfilerStop ());
    exit (0);
  }
}

int main()
{
  init ();
  printf ("ptr is %p\n", (void *)ptr);

  gpuErrchk (cudaProfilerStart ());

  GPU_DATA_TYPE *d_a;
  gpuErrchk (cudaMalloc (&d_a, sizeof(GPU_DATA_TYPE) * GPU_SIZE));

  init_kernel<<<128,1024>>> (d_a);

  Sender *sender = new Sender(shmid, ptr, d_a);

  cudaIpcMemHandle_t handle;
  gpuErrchk (cudaIpcGetMemHandle (&handle, d_a));
  
  memcpy (&(ptr->memHandle), &handle, sizeof (cudaIpcMemHandle_t));

  sender->set_GPUIPC_handle (&handle);

  // set the signal handling function
  if (signal (SIGINT, sigInt_handler) == SIG_ERR)
  {
    printf ("cannot handle SIGINT\n");
    exit(-1);
  }

  // main loop
  while (true)
  // for (int i = 0;i < 1000; i++)
  {
    sender->update ();
    sender->wait ();
    sender->process ();
    sender->notify ();
    // usleep (20);
  }
}
