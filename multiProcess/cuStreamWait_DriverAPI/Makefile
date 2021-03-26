CC = nvcc

PTX = -ptx

all: sender pluser

sender: sender.o
	$(CC) $^ -o $@ -lcuda

pluser: pluser.o
	$(CC) $^ -o $@ -lcuda

sender.o: sender.cpp sender_init.ptx sender_notify.ptx sender_process.ptx sender_wait.ptx
	$(CC) $< -c -o $@

sender_init.ptx: sender_init.cu
	$(CC) $^ $(PTX) -o $@

sender_wait.ptx: sender_wait.cu
	$(CC) $^ $(PTX) -o $@

sender_process.ptx: sender_process.cu
	$(CC) $^ $(PTX) -o $@

sender_notify.ptx: sender_notify.cu
	$(CC) $^ $(PTX) -o $@

pluser.o: pluser.cpp pluser_notify.ptx pluser_process.ptx pluser_wait.ptx
	$(CC) $< -c -o $@

pluser_wait.ptx: pluser_wait.cu
	$(CC) $^ $(PTX) -o $@

pluser_process.ptx: pluser_process.cu
	$(CC) $^ $(PTX) -o $@

pluser_notify.ptx: pluser_notify.cu
	$(CC) $^ $(PTX) -o $@

