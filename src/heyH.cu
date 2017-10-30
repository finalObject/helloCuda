#include <stdlib.h>  
#include <cuda_runtime.h> 
#include <stdio.h>  
#include "heyD.cu"
void display(char*,int);
__global__ void cvlUnit(const char *,const char *,const char*,const char *,char *,char *,char*,int,int,int);  
void initCvlUnit(char* imgRH,char* imgGH,char* imgBH,char* coreH,int lenX,int lenY,int lenCore,char* imgRD,char*imgGD,char* imgBD,char* coreD,char* outRD,char* outGD,char* outBD);
//主机端代码
int func() // 注意这里定义形式  
{

	int lenX=10;
	int lenY = 10;
	int lenCore = 3;
	int sizeOfImg = lenX*lenY*sizeof(char);
	int sizeOfCore = lenCore*lenCore*sizeof(char);

	//cuda malloc have to be done first,but i do not know why
	char *imgRD=NULL,*imgGD=NULL,*imgBD=NULL,*coreD=NULL,*outRD=NULL,*outGD=NULL,*outBD=NULL;
	cudaMalloc((void**)&imgRD,sizeOfImg);
	cudaMalloc((void**)&imgGD,sizeOfImg);
	cudaMalloc((void**)&imgBD,sizeOfImg);
	cudaMalloc((void**)&coreD,sizeOfCore);
	cudaMalloc((void**)&outRD,sizeOfImg);
	cudaMalloc((void**)&outGD,sizeOfImg);
	cudaMalloc((void**)&outBD,sizeOfImg);


	//host data define
	char *imgRH=(char*)malloc(sizeOfImg);
	char *imgGH=(char*)malloc(sizeOfImg);
	char *imgBH=(char*)malloc(sizeOfImg);
	char *coreH=(char*)malloc(sizeOfCore);
	int i,j;
	for(i=0;i<lenX;i++){
		for(j=0;j<lenY;j++){
			coreH[j*lenX+i]=1;
			imgRH[j*lenX+i]=(rand()%128);
			imgGH[j*lenX+i]=(rand()%128);
			imgBH[j*lenX+i]=(rand()%128);
		}
	}
	initCvlUnit(imgRH,imgGH,imgBH,coreH,lenX,lenY,lenCore,imgRD,imgGD,imgBD,coreD,outRD,outGD,outBD);
	
	return 0;
}
void initCvlUnit(char* imgRH,char* imgGH,char* imgBH,char* coreH,int lenX,int lenY,int lenCore,char* imgRD,char*imgGD,char* imgBD,char* coreD,char* outRD,char* outGD,char* outBD){
	int sizeOfImg = lenX*lenY*sizeof(char);
	int sizeOfCore = lenCore*lenCore*sizeof(char);
	char *outRH=(char*)malloc(sizeOfImg);
	char *outGH=(char*)malloc(sizeOfImg);
	char *outBH=(char*)malloc(sizeOfImg);
	cudaMemcpy(imgRD,imgRH,sizeOfImg,cudaMemcpyHostToDevice);
	cudaMemcpy(imgGD,imgGH,sizeOfImg,cudaMemcpyHostToDevice);
	cudaMemcpy(imgBD,imgBH,sizeOfImg,cudaMemcpyHostToDevice);
	cudaMemcpy(coreD,coreH,sizeOfCore,cudaMemcpyHostToDevice);

	int lenBlock =3;
	int lenGridX = lenX/lenBlock;
	int lenGridY = lenY/lenBlock;
	if(lenBlock*lenGridX!=lenX)lenGridX++;
	if(lenBlock*lenGridY!=lenX)lenGridY++;
	dim3 dimGrid(lenGridX,lenGridY);
	dim3 dimBlock(lenBlock,lenBlock);
	printf("  r:\n");display(imgRH,lenX);
	printf("  g:\n");display(imgGH,lenX);
	printf("  b:\n");display(imgBH,lenX);
	printf("  core:\n");display(coreH,lenCore);
	printf("  %d,%d,%d,%d,%d,%d,%d\n",lenGridX,lenGridY,lenBlock,lenBlock,lenX,lenY,lenCore);
	cvlUnit<<<dimGrid,dimBlock>>>(imgRD,imgGD,imgBD,coreD,outRD,outGD,outBD,lenX,lenY,lenCore);

	cudaMemcpy(outRH,outRD,sizeOfImg,cudaMemcpyDeviceToHost);
	cudaMemcpy(outGH,outGD,sizeOfImg,cudaMemcpyDeviceToHost);
	cudaMemcpy(outBH,outBD,sizeOfImg,cudaMemcpyDeviceToHost);
	printf("imgRH:\n");display(imgRH,lenX);
	printf("imgGH:\n");display(imgGH,lenX);
	printf("imgBH:\n");display(imgBH,lenX);
	printf("coreH:\n");display(coreH,lenCore);
	printf("outRH:\n");display(outRH,lenX);
	printf("outGH:\n");display(outGH,lenX);
	printf("outBH:\n");display(outBH,lenX);

}
void display(char* a,int num){
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







