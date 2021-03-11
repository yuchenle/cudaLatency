CC = nvcc


all: cudaDeviceSync cudaStreamSync cudaEventSync cudaGraphSync cudaSameStream

cudaDeviceSync: cudaDeviceSync.cu
	$(CC) $^ -o $@

cudaStreamSync: cudaStreamSync.cu
	$(CC) $^ -o $@

cudaEventSync: cudaEventSync.cu
	$(CC) $^ -o $@

cudaGraphSync: cudaGraphSync.cu
	$(CC) $^ -o $@

cudaSameStream: cudaSameStream.cu
	$(CC) $^ -o $@

clean:
	rm cudaDeviceSync cudaStreamSync cudaEventSync cudaGraphSync cudaSameStream

