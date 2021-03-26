#ifndef SENDER_H
#define SENDER_H

#include "MicroUnit.hpp"

class Sender: public MicroUnit
{
  public:

  // inheriting Constructor && Destructor
  Sender ();
  Sender (int, TYPE *, GPU_DATA_TYPE *);
  ~Sender ();

  // pipeline functions
  void update ();
  void wait ();
  void process ();
  void notify ();
  void kernel_wrapper ();

  // GETTERS
  int get_SHM_id ();
  TYPE *get_SHM_ptr ();
  cudaIpcMemHandle_t *get_GPUIPC_handle ();
  GPU_DATA_TYPE *get_d_data();

  // SETTERS
  void set_SHM_id (int);
  void set_SHM_ptr (TYPE *);
  void set_GPUIPC_handle (cudaIpcMemHandle_t *);
  void set_d_data (GPU_DATA_TYPE *);

  protected:

  cudaIpcMemHandle_t *mem_handle;
  GPU_DATA_TYPE *d_data;
};

#endif //SENDER_H
