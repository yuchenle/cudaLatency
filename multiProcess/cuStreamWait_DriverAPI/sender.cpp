#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <string.h>

// CUDA
#include <cuda.h>
#include <builtin_types.h>

#include "./const.h"

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
CUfunction init_function;
CUfunction wait_function;
CUfunction process_function;
CUfunction notify_function;
CUmodule init_module;
CUmodule wait_module;
CUmodule process_module;
CUmodule notify_module;
size_t totalGlobalMem;

char *sender_process_file = (char *)"sender_process.ptx";
char *sender_process =(char *)"sender_process";
char *sender_wait_file = (char *)"sender_wait.ptx";
char *sender_wait = (char *)"sender_wait";
char *sender_notify_file = (char *)"sender_notify.ptx";
char *sender_notify = (char *)"sender_notify";
char *sender_init_file = (char *)"sender_init.ptx";
char *sender_init = (char *)"sender_init";

void init_shm (TYPE *ptr)
{
  for (int i = 0; i < SIZE; i++)
    ptr->ready[i] = WAIT_VALUE;
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
  err = cuModuleLoad(&wait_module, sender_wait_file );
  if (err != CUDA_SUCCESS) {
      fprintf(stderr, "* Error loading the module %s\n", sender_wait_file);
      cuCtxDetach(context);
      exit(-1);
  }

  // getting process module
  err = cuModuleLoad(&process_module, sender_process_file );
  if (err != CUDA_SUCCESS) {
      fprintf(stderr, "* Error loading the module %s\n", sender_process_file);
      cuCtxDetach(context);
      exit(-1);
  }

  // getting notify module
  err = cuModuleLoad(&notify_module, sender_notify_file );
  if (err != CUDA_SUCCESS) {
      fprintf(stderr, "* Error loading the module %s\n", sender_wait_file);
      cuCtxDetach(context);
      exit(-1);
  }

  // getting init module
  err = cuModuleLoad(&init_module, sender_init_file );
  if (err != CUDA_SUCCESS) {
      fprintf(stderr, "* Error loading the module %s\n", sender_init_file);
      cuCtxDetach(context);
      exit(-1);
  }

  // getting wait kernel handle
  err = cuModuleGetFunction(&wait_function, wait_module, sender_wait);
  if (err != CUDA_SUCCESS) {
      fprintf(stderr, "* Error getting kernel function %s\n", sender_wait);
      cuCtxDetach(context);
      exit(-1);
  }

  // getting process kernel handle
  err = cuModuleGetFunction(&process_function, process_module, sender_process);
  if (err != CUDA_SUCCESS) {
      fprintf(stderr, "* Error getting kernel function %s\n", sender_process);
      cuCtxDetach(context);
      exit(-1);
  }

  // getting notify kernel handle
  err = cuModuleGetFunction(&notify_function, notify_module, sender_notify);
  if (err != CUDA_SUCCESS) {
      fprintf(stderr, "* Error getting kernel function %s\n", sender_notify);
      cuCtxDetach(context);
      exit(-1);
  }

  // getting init kernel handle
  err = cuModuleGetFunction(&init_function, init_module, sender_init);
  if (err != CUDA_SUCCESS) {
      fprintf(stderr, "* Error getting kernel function %s\n", sender_init);
      cuCtxDetach(context);
      exit(-1);
  }
}

int main()
{
  init();
  initCuda();


  // allocating device data pointer
  CUdeviceptr d_a;
  checkCudaErrors (cuMemAlloc (&d_a, sizeof(GPU_DATA_TYPE) * GPU_SIZE));

  // calling initialization kernel to initialize device data
  void *args[] = {&d_a};
  checkCudaErrors (cuLaunchKernel (init_function, 128, 1, 1,
                                   1024, 1, 1,
                                   0,
                                   0, args, NULL));

  checkCudaErrors (cuStreamWriteValue32 (0, d_a, 1, 0x0));
  checkCudaErrors (cuCtxSynchronize());
  // get memHandle of the device data to be shared
  CUipcMemHandle handle;
  checkCudaErrors (cuIpcGetMemHandle (&handle, d_a));
  memcpy (&(ptr->memHandle), &handle, sizeof (CUipcMemHandle));

  //while (true)
  for (int i = 0 ; i < 1000000; i++)
  {
    // printf ("from sender\n");

    // wait   
    // checkCudaErrors (cuLaunchKernel (wait_function, 1, 1, 1,
    //                                  1, 1, 1,
    //                                  0,
    //                                  NULL, args, NULL));

    // block until d_a == 1
    checkCudaErrors (cuStreamWaitValue32 (0, d_a, 0, 0x1));

    // process, by 128 blocks of 1024 threads, arbitrary numbers
    checkCudaErrors (cuLaunchKernel (process_function, 128, 1, 1,
                                     1024, 1, 1,
                                     0,
                                     0, args, NULL));

    // notify
    checkCudaErrors (cuStreamWriteValue32 (0, d_a, 1, 0x0));
    // checkCudaErrors (cuLaunchKernel (notify_function, 1, 1, 1,
    //                                  1, 1, 1,
    //                                  0,
    //                                  0, args, NULL));
    // checkCudaErrors (cuCtxSynchronize());
    // printf (" sender, iteration %d\n", i);
  }
}

