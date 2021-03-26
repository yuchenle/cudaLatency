#ifndef CONST_H
#define CONST_H

#define GPU_SIZE 1024*4 // 2^22
#define GPU_DATA_TYPE float

#define SIZE 2
#define WAIT_VALUE false // false: data not ready, true: data ready to read
#define TYPE struct SharedData
#define FILENAME "/tmp"
#define FILEID 666

struct SharedData
{
  bool ready[SIZE];
  cudaStream_t RT_stream;
  cudaStream_t callBackStream;
  cudaIpcMemHandle_t memHandle;
  cudaIpcEventHandle_t eventHandle;
};

#if 0 // Not used by now
#if defined(__CUDACC__) //NVCC
  #define MY_ALIGN(n) __align__(n)
#elif defined(__GNUC__) //GCC
  #define MY_ALIGN(n) __attribute__((aligned(n))
#else
  #error "Define memory alignment for your compiled"
#endif
#endif //0

#endif
