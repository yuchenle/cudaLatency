#include <stdio.h>
#include <stdlib.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/types.h>
#include <unistd.h>
#include <errno.h>
#include <signal.h>
#include <cuda_profiler_api.h>

#include "../cudaErr.h"
#include "const.h"
#include "Pluser.hpp"
#include "timer.h"

//global variables to be initialized
int shmid;
TYPE *ptr;

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
  shmid = shmget (key, sizeof (TYPE), 0666|IPC_CREAT);
  if (shmid == -1)
  {
    printf ("shmget failed, errno = %s\n", strerror(errno));
    exit (-1);
  }

  // attach the SHM to this process
  ptr = (TYPE *) shmat (shmid, (void *)0, 0);
}

__global__ void print_first (GPU_DATA_TYPE *data)
{
  printf ("before main loop, first element is %.4f\n", data[0]);
}

void sigInt_handler (int sig)
{
  if (sig == SIGINT)
  {
    printf ("pluser received SIGINT, calling cudaProfilerStop before exiting \n");
    gpuErrchk (cudaProfilerStop ());
    exit (0);
  }
}

int main()
{
  // a previous process should have allocated the SHM, we attach to it
  init ();
  printf ("ptr is %p\n", (void *)ptr);

  gpuErrchk (cudaProfilerStart ());

  Pluser *pluser = new Pluser (shmid, ptr);

  pluser->set_GPUIPC_handle (&(ptr->memHandle));
  printf ("init pluser: setting d_data\n");
  GPU_DATA_TYPE *addr_d_data;

  gpuErrchk (cudaIpcOpenMemHandle ((void **)&addr_d_data, *(pluser->get_GPUIPC_handle()), cudaIpcMemLazyEnablePeerAccess));
  pluser->set_d_data (addr_d_data);

  // set the signal handling function
  if (signal (SIGINT, sigInt_handler) == SIG_ERR)
  {
    printf ("cannot handle SIGINT\n");
    exit(-1);
  }

  // main loop
  while (true)
  // for (int i = 0; i < 1000; i++)
  {
    struct timespec stamp, previous_stamp;
    clock_gettime (CLOCK_MONOTONIC, &previous_stamp);
    double wtime;

    pluser->update ();
    pluser->wait ();
    pluser->process ();
    pluser->notify ();

    clock_gettime (CLOCK_MONOTONIC, &stamp);
    wtime = (stamp.tv_sec - previous_stamp.tv_sec) * 1000000 + (stamp.tv_nsec - previous_stamp.tv_nsec) / 1000;
    printf ("1 iteration took %.4f us\n", wtime);
    // usleep (20);
  }
  // gpuErrchk (cudaDeviceSynchronize ());
}
