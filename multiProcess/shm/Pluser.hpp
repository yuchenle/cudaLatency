#ifndef PLUSER_H
#define PLUSER_H

#include "MicroUnit.hpp"

class Pluser: public MicroUnit
{
  public:

  // inheriting Constructor && Destructor
  Pluser ();
  Pluser (int, TYPE *);
  ~Pluser ();

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

#endif //PLUSER_H
