#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <vector>

#include "../cudaErr.h"

__global__ void dummy_kernel()
{
}

int main()
{
  cudaGraph_t graph;
  cudaStream_t streamForGraph;
  gpuErrchk (cudaGraphCreate (&graph, 0));
  gpuErrchk (cudaStreamCreate (&streamForGraph));

  cudaGraphNode_t dummyNode1, dummyNode2;

  std::vector<cudaGraphNode_t> nodeDep;

  // adding first dummy node to the graph
  cudaKernelNodeParams kernelNodeParams = {0};

  kernelNodeParams.func = (void *) dummy_kernel;
  kernelNodeParams.gridDim = dim3 (1, 1, 1); 
  kernelNodeParams.blockDim = dim3 (1, 1, 1); 
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.extra = NULL;
  kernelNodeParams.kernelParams = (void **)NULL;
  gpuErrchk (cudaGraphAddKernelNode (&dummyNode1, graph, NULL, 0, &kernelNodeParams));

  nodeDep.push_back (dummyNode1);

  // adding second dummy node
  // memset (&kernelNodeParams, 0, sizeof (kernelNodeParams));
  // kernelNodeParams.func = (void *)dummy_kernel;
  // kernelNodeParams.gridDim = dim3 (1,1,1);
  // kernelNodeParams.blockDim = dim3 (1,1,1);
  // kernelNodeParams.sharedMemBytes = 0;
  // kernelNodeParams.extra = NULL;
  gpuErrchk (
	cudaGraphAddKernelNode (&dummyNode2, graph, nodeDep.data(), 
				nodeDep.size(), &kernelNodeParams));

  cudaGraphNode_t *nodes = NULL;
  size_t numNodes = 0;
  gpuErrchk (cudaGraphGetNodes (graph, nodes, &numNodes));
  printf ("number of nodes in graph is %zu\n", numNodes);

  cudaGraphExec_t graphExec;
  gpuErrchk (cudaGraphInstantiate (&graphExec, graph, NULL, NULL, 0));

  struct timespec stamp, previous_stamp;
  clock_gettime (CLOCK_MONOTONIC, &stamp);
  double wtime;
  
  for (int i = 0; i < 1000000; i++)
  {
    gpuErrchk (cudaGraphLaunch (graphExec, streamForGraph));
    gpuErrchk (cudaStreamSynchronize (streamForGraph));
    memcpy (&previous_stamp, &stamp, sizeof (struct timespec));
    clock_gettime (CLOCK_MONOTONIC, &stamp);
    wtime = (stamp.tv_sec - previous_stamp.tv_sec) * 1000000 + (stamp.tv_nsec - previous_stamp.tv_nsec) / 1000;
    printf ("%.4f \n", wtime);
  }
  
}
