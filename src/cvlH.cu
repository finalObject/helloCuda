//host code
#include <stdlib.h>  
#include <cuda_runtime.h> 
#include <stdio.h>  
#include <opencv2/opencv.hpp>
#include "cvlD.cu"
using namespace cv;
__global__ void cvlUnit(const char *imgR,const char *imgG,const char *imgB,const char *core,char *outR,char *outG,char *outB,int lenX,int lenY,int lenCore);  
void initCvlUnit(char* imgRH,char* imgGH,char* imgBH,char* coreH,int lenX,int lenY,int lenCore,char* imgRD,char*imgGD,char* imgBD,char* coreD,char* outRD,char* outGD,char* outBD);
void mat2pointerCore(Mat core,char* coreH);
void mat2pointerImg(Mat img,int z,char* imgH);
void display(char*,int);
Mat pointerToMat(char* r,char* g,char* b,Mat img);
//机端代码
Mat cudaCvl(Mat img,Mat core){
	int lenX = img.cols;
	int lenY = img.rows; 
	int lenCore = core.cols;
	
//	printf("%d,%d,%d\n",lenX,lenY,lenGridX);
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
	char* imgRH = (char*)malloc(sizeOfImg);
	char* imgGH = (char*)malloc(sizeOfImg);
	char* imgBH = (char*)malloc(sizeOfImg);
	char* coreH = (char*)malloc(sizeOfCore);
	mat2pointerImg(img,0,imgBH);
	mat2pointerImg(img,1,imgGH);
	mat2pointerImg(img,2,imgRH);
	mat2pointerCore(core,coreH);


	initCvlUnit(imgRH,imgGH,imgBH,coreH,lenX,lenY,lenCore,imgRD,imgGD,imgBD,coreD,outRD,outGD,outBD);

	Mat result = pointerToMat(imgRH,imgGH,imgBH,img);
	return result;
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

	int lenBlock =16;
	int lenGridX = lenX/lenBlock;
	int lenGridY = lenY/lenBlock;
	if(lenBlock*lenGridX!=lenX)lenGridX++;
	if(lenBlock*lenGridY!=lenX)lenGridY++;
	dim3 dimGrid(lenGridX,lenGridY);
	dim3 dimBlock(lenBlock,lenBlock);
//	printf("  r:\n");display(imgRH,lenX);
//	printf("  g:\n");display(imgGH,lenX);
//	printf("  b:\n");display(imgBH,lenX);
//	printf("  core:\n");display(coreH,lenCore);
//	printf("  %d,%d,%d,%d,%d,%d,%d\n",lenGridX,lenGridY,lenBlock,lenBlock,lenX,lenY,lenCore);
	cvlUnit<<<dimGrid,dimBlock>>>(imgRD,imgGD,imgBD,coreD,outRD,outGD,outBD,lenX,lenY,lenCore);

	cudaMemcpy(outRH,outRD,sizeOfImg,cudaMemcpyDeviceToHost);
	cudaMemcpy(outGH,outGD,sizeOfImg,cudaMemcpyDeviceToHost);
	cudaMemcpy(outBH,outBD,sizeOfImg,cudaMemcpyDeviceToHost);
//	printf("imgRH:\n");display(imgRH,lenX);
//	printf("imgGH:\n");display(imgGH,lenX);
//	printf("imgBH:\n");display(imgBH,lenX);
//	printf("coreH:\n");display(coreH,lenCore);
//	printf("outRH:\n");display(outRH,lenX);
//	printf("outGH:\n");display(outGH,lenX);
//	printf("outBH:\n");display(outBH,lenX);

	cudaMemcpy(imgRH,outRH,sizeOfImg,cudaMemcpyHostToHost);
	cudaMemcpy(imgGH,outGH,sizeOfImg,cudaMemcpyHostToHost);
	cudaMemcpy(imgBH,outBH,sizeOfImg,cudaMemcpyHostToHost);
	return;
}
Mat pointerToMat(char* r,char* g,char* b,Mat img){
	Mat result = img.clone();
	int lenX = img.cols;
	int lenY = img.rows;
	int i,j;
	for(i=0;i<lenX;i++){
		for(j=0;j<lenY;j++){
			result.at<Vec3b>(i,j)[0]=b[i*lenX+j];
			result.at<Vec3b>(i,j)[1]=g[i*lenX+j];
			result.at<Vec3b>(i,j)[2]=r[i*lenX+j];
		}
	}
	return result;
}
void mat2pointerImg(Mat img,int z,char* imgH){
	int lenX = img.cols;
	int lenY = img.rows;
	int i,j;
	for(i=0;i<lenX;i++){
		for(j=0;j<lenY;j++){
			imgH[i*lenX+j]=img.at<Vec3b>(i,j)[z];
		}
	}
	return;
}
void mat2pointerCore(Mat core,char* coreH){
	int lenCore = core.rows;
	int i,j;
	for(i=0;i<lenCore;i++){
		for(j=0;j<lenCore;j++){
			coreH[i*lenCore+j]=core.at<char>(i,j);
		}		
	}
	return;
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






