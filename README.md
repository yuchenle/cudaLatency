# cudaLatency
The objective of this project is to measure the cuda synchronization costs.
Two different scenarios are considered:

## Single process
Cases where different kernels submitting to different CUDA streams or to the same CUDA stream. When using cudaDeviceSynchronize() or cudaEvent_t to synchronize.

## Multiple process
Two CPU processes submitting jobs to the CPU, operating the same GPU data. We synchronize them with SHM (as semaphore), GPU active waiting and cuStreamWaitValue (driver API)

# Always launching the sender first
