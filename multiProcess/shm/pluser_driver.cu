#include <stdio.h>
#include <stdlib.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/types.h>
#include <unistd.h>
#include <errno.h>

#include "../cudaErr.h"
#include "const.h"
#include "Pluser.hpp"
#include "timer.h"
//global variables to be initialized
// key_t key;
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
  shmid = shmget (key, sizeof (TYPE) * SIZE, 0666|IPC_CREAT);
  if (shmid == -1)
  {
    printf ("shmget failed, errno = %s\n", strerror(errno));
    exit (-1);
  }

  // attach the SHM to this process
  ptr = (TYPE *) shmat (shmid, (void *)0, 0);
  * (ptr + (SIZE - 1)) = WAIT_VALUE;
}

int main()
{
  init ();
  printf ("ptr is %p\n", (void *)ptr);

  Pluser *pluser = new Pluser (shmid, ptr);

  struct timespec stamp, previous_stamp;
  double wtime;

  clock_gettime (CLOCK_MONOTONIC, &previous_stamp);

  // main loop
  // while (true)
  for (int i = 0; i < 1000000; i++)
  {

    pluser->update ();
    pluser->wait ();
    pluser->process ();

    // timer stop at iteration i+1
    clock_gettime (CLOCK_MONOTONIC, &stamp);
    wtime = (stamp.tv_sec - previous_stamp.tv_sec) * 1000000000 + (stamp.tv_nsec - previous_stamp.tv_nsec);
    printf ("1 iteration took %.4f us\n", wtime);

    // timer starts  at iteration i
    clock_gettime (CLOCK_MONOTONIC, &previous_stamp);

    pluser->notify ();
    //usleep (20);
  }
}
