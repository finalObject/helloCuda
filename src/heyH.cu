#include <stdlib.h>  
#include <cuda_runtime.h> 
#include <stdio.h>  
#include "heyD.cu"
void display(int*,int);
__global__ void matrixAdd(const int *, const int *, int *, int);  
//主机端代码
int func() // 注意这里定义形式  
{
	int num = 4;
	int size = num*num*sizeof(int);
	int *matrixA=(int*)malloc(size);
	int *matrixB=(int*)malloc(size);
	int i=0;
	for (i=0;i<num*num;i++){
		matrixA[i]=i+1;
		matrixB[i]=2*(i+1);
	}
	int *matrixC=(int*)malloc(size);

	int *matrixAd=NULL,*matrixBd=NULL,*matrixCd=NULL;

	cudaMalloc((void**)&matrixAd,size);
	cudaMalloc((void**)&matrixBd,size);
	cudaMalloc((void**)&matrixCd,size);
	cudaMemcpy(matrixAd,matrixA,size,cudaMemcpyHostToDevice);
	cudaMemcpy(matrixBd,matrixB,size,cudaMemcpyHostToDevice);

	int blockX = 2;int blockY = 2;
	dim3 dimGrid(num/blockX,num/blockY);
	dim3 dimBlock(blockX,blockY);
	matrixAdd<<<dimGrid,dimBlock>>>(matrixAd,matrixBd,matrixCd,num);

	cudaMemcpy(matrixC,matrixCd,size,cudaMemcpyDeviceToHost);
	printf("matrixA:\n");display(matrixA,num);
	printf("matrixB:\n");display(matrixB,num);
	printf("matrixC:\n");display(matrixC,num);
	return 0;
}
void display(int* a,int num){
	int i=0,j=0;
	int index=0;
	for(i=0;i<num;i++){
		for(j=0;j<num;j++){
			index=i*num+j;
			printf("\t%2d",a[index]);
		}
		printf("\n");
	}
}







