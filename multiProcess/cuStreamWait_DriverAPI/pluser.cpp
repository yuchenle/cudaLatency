#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <string.h>
#include <time.h>

// CUDA
#include <cuda.h>
#include <builtin_types.h>

extern "C" 
{
  #include "const.h"
}

#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)

inline void __checkCudaErrors( CUresult err, const char *file, const int line )
{
    if( CUDA_SUCCESS != err) {
        fprintf(stderr,
                "CUDA Driver API error = %04d from file <%s>, line %i.\n",
                err, file, line );
        exit(-1);
    }   
}

// global variables to be initialized
int shmid;
TYPE *ptr;
cudaStream_t RT_stream_g;
cudaStream_t callBack_stream_g;
cudaEvent_t event_g;

CUdevice device;
CUcontext context;
CUfunction wait_function;
CUfunction process_function;
CUfunction notify_function;
CUmodule wait_module;
CUmodule process_module;
CUmodule notify_module;
size_t totalGlobalMem;

char *pluser_process_file = (char *)"pluser_process.ptx";
char *pluser_process =(char *)"pluser_process";
char *pluser_wait_file = (char *)"pluser_wait.ptx";
char *pluser_wait = (char *)"pluser_wait";
char *pluser_notify_file = (char *)"pluser_notify.ptx";
char *pluser_notify = (char *)"pluser_notify";

__device__ CUdeviceptr device_a;

void init ()
{
  // FILE to key
  key_t key = ftok (FILENAME, FILEID);
  if (key == -1)
  {
    printf ("ftok failed, errno = %s\n", strerror(errno));
    exit (-1);
  }

  shmid = shmget (key, sizeof (TYPE), 0666|IPC_CREAT);
  if (shmid == -1)
  {
    printf ("shmget failed, errno = %s\n", strerror(errno));
    exit (-1);
  }

  // attach the SHM to this process
  ptr = (TYPE *) shmat (shmid, (void *)0, 0);
}


void initCuda()
{
  int deviceCount = 0;
  CUresult err = cuInit(0);
  int major = 0, minor = 0;

  if (err == CUDA_SUCCESS)
      checkCudaErrors(cuDeviceGetCount(&deviceCount));

  if (deviceCount == 0) {
      fprintf(stderr, "Error: no devices supporting CUDA\n");
      exit(-1);
  }

  // get first CUDA device
  checkCudaErrors(cuDeviceGet(&device, 0));
  char name[100];
  cuDeviceGetName(name, 100, device);
  printf("> Using device 0: %s\n", name);

  // get compute capabilities and the devicename
  checkCudaErrors( cuDeviceComputeCapability(&major, &minor, device) );
  printf("> GPU Device has SM %d.%d compute capability\n", major, minor);

  checkCudaErrors( cuDeviceTotalMem(&totalGlobalMem, device) );
  printf("  Total amount of global memory:   %llu bytes\n",
         (unsigned long long)totalGlobalMem);
  printf("  64-bit Memory Address:           %s\n",
         (totalGlobalMem > (unsigned long long)4*1024*1024*1024L)?
         "YES" : "NO");

  // get device 0
  err = cuCtxCreate(&context, 0, device);
  if (err != CUDA_SUCCESS) {
      fprintf(stderr, "* Error initializing the CUDA context.\n");
      cuCtxDetach(context);
      exit(-1);
  }

  // getting wait module
  err = cuModuleLoad(&wait_module, pluser_wait_file );
  if (err != CUDA_SUCCESS) {
      fprintf(stderr, "* Error loading the module %s\n", pluser_wait_file);
      cuCtxDetach(context);
      exit(-1);
  }

  // getting process module
  err = cuModuleLoad(&process_module, pluser_process_file );
  if (err != CUDA_SUCCESS) {
      fprintf(stderr, "* Error loading the module %s\n", pluser_process_file);
      cuCtxDetach(context);
      exit(-1);
  }

  // getting notify module
  err = cuModuleLoad(&notify_module, pluser_notify_file );
  if (err != CUDA_SUCCESS) {
      fprintf(stderr, "* Error loading the module %s\n", pluser_wait_file);
      cuCtxDetach(context);
      exit(-1);
  }

  // getting wait kernel handle
  err = cuModuleGetFunction(&wait_function, wait_module, pluser_wait);
  if (err != CUDA_SUCCESS) {
      fprintf(stderr, "* Error getting kernel function %s\n", pluser_wait);
      cuCtxDetach(context);
      exit(-1);
  }

  // getting process kernel handle
  err = cuModuleGetFunction(&process_function, process_module, pluser_process);
  if (err != CUDA_SUCCESS) {
      fprintf(stderr, "* Error getting kernel function %s\n", pluser_process);
      cuCtxDetach(context);
      exit(-1);
  }

  // getting notify kernel handle
  err = cuModuleGetFunction(&notify_function, notify_module, pluser_notify);
  if (err != CUDA_SUCCESS) {
      fprintf(stderr, "* Error getting kernel function %s\n", pluser_notify);
      cuCtxDetach(context);
      exit(-1);
  }

}

int main()
{
  init();
  initCuda();


  CUdeviceptr d_a;
  checkCudaErrors (cuIpcOpenMemHandle (&d_a, ptr->memHandle, CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS));
  device_a = d_a;

  void *args[] = {&d_a};
  // while (true)
  for (int i = 0 ; i < 1000000; i++)
  {

    struct timespec stamp, previous_stamp;
    clock_gettime(CLOCK_MONOTONIC, &previous_stamp);
    double wtime;

    // printf ("from pluser\n");
    // wait   
    // checkCudaErrors (cuLaunchKernel (wait_function, 1, 1, 1,
    //                                  1, 1, 1,
    //                                  0,
    //                                  NULL, args, NULL));

    // block until d_a == 0
    checkCudaErrors (cuStreamWaitValue32 (0, d_a, 1, 0x1));

    // process, 128 blocks of 1024 threads, these values are set arbitrarily
    checkCudaErrors (cuLaunchKernel (process_function, 128, 1, 1,
                                     1024, 1, 1,
                                     0,
                                     0, args, NULL));

    // notify
    checkCudaErrors (cuStreamWriteValue32 (0, d_a, 0, 0x0));
    // checkCudaErrors (cuLaunchKernel (notify_function, 1, 1, 1,
    //                                   1, 1, 1,
    //                                   0,
    //                                   0, args, NULL));
    
    clock_gettime(CLOCK_MONOTONIC, &stamp);
    wtime = (stamp.tv_sec - previous_stamp.tv_sec) * 1000000000 + (stamp.tv_nsec - previous_stamp.tv_nsec);
    printf ("one iteration took %.4f\n", wtime);
    // checkCudaErrors (cuCtxSynchronize());
  }
}

