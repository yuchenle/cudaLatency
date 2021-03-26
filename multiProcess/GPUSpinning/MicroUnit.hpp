#ifndef MICROUNIT_H
#define MICROUNIT_H

// This class is an abstract class (interface)
// calling new MicroUnit is not correct.

extern "C"
{
 #include "const.h"
}

using namespace std;

class MicroUnit
{
  public:

  // GETTER
  virtual int get_SHM_id () = 0;
  virtual TYPE *get_SHM_ptr () = 0;

  // SETTER
  virtual void set_SHM_id (int) = 0;
  virtual void set_SHM_ptr (TYPE *) = 0;

  // pipeline functions
  virtual void update () = 0;
  virtual void wait () = 0;
  virtual void process () = 0;
  virtual void notify () = 0;

  protected:

  int shmid;
  TYPE *ptr;
};

#endif //MICROUNIT_H
