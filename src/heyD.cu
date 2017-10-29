#include <stdlib.h>  
#include <cuda_runtime.h> 
#include <stdio.h>  

//设备端代码
__global__ void matrixAdd(const int *A, const int *B, int *C, int num)  
{  
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	C[x * num + y] = A[x * num + y] + B[x * num + y];
	return;
}  
