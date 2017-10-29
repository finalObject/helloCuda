//host code
#include <stdlib.h>  
#include <cuda_runtime.h> 
#include <stdio.h>  
#include <opencv2/opencv.hpp>
#include "cvlD.cu"
using namespace cv;
__global__ void cvlUnit(const char *imgR,const char *imgG,const char *imgB,const char *core,char *outR,char *outG,char *outB,int lenX,int lenY,int lenCore);  
void mat2pointerCore(Mat core,char* coreH);
void mat2pointerImg(Mat img,int z,char* imgH);
Mat pointerToMat(char* r,char* g,char* b,Mat img);
//机端代码
Mat cudaCvl(Mat img,Mat core){
	int lenX = img.cols;
	int lenY = img.rows; 
	int lenCore = core.cols;
	int lenBlock = 16;
	int lenGridX = lenX/lenBlock;
	int lenGridY = lenY/lenBlock;
	if(lenBlock*lenGridX!=lenX)lenGridX++;
	if(lenBlock*lenGridY!=lenX)lenGridY++;

	int sizeOfImage = lenX*lenY*sizeof(char);
	int sizeOfCore = lenCore*lenCore*sizeof(char);

	//host data define
	char* imgRH = (char*)malloc(sizeOfImage);
	char* imgGH = (char*)malloc(sizeOfImage);
	char* imgBH = (char*)malloc(sizeOfImage);
	char* coreH = (char*)malloc(sizeOfCore);
	mat2pointerImg(img,0,imgBH);
	mat2pointerImg(img,1,imgGH);
	mat2pointerImg(img,2,imgRH);
	mat2pointerCore(core,coreH);

	//device data define
	char *imgRD=NULL,*imgGD=NULL,*imgBD=NULL,*coreD=NULL;
	cudaMalloc((void**)&imgRD,sizeOfImage);
	cudaMalloc((void**)&imgGD,sizeOfImage);
	cudaMalloc((void**)&imgBD,sizeOfImage);
	cudaMalloc((void**)&coreD,sizeOfCore);
	cudaMemcpy(imgRD,imgRH,sizeOfImage,cudaMemcpyHostToDevice);
	cudaMemcpy(imgGD,imgGH,sizeOfImage,cudaMemcpyHostToDevice);
	cudaMemcpy(imgBD,imgBH,sizeOfImage,cudaMemcpyHostToDevice);
	cudaMemcpy(coreD,coreH,sizeOfCore,cudaMemcpyHostToDevice);
	char *outRD=NULL,*outGD=NULL,*outBD=NULL;
	cudaMalloc((void**)&outRD,sizeOfImage);
	cudaMalloc((void**)&outRD,sizeOfImage);
	cudaMalloc((void**)&outRD,sizeOfImage);

	//start work
	dim3 dimGrid(lenGridX,lenGridY);
	dim3 dimBlock(lenBlock,lenBlock);
	cvlUnit<<<dimGrid,dimBlock>>>(imgRD,imgGD,imgBD,coreD,outRD,outGD,outBD,lenX,lenY,lenCore);


	cudaMemcpy(imgRH,outRD,sizeOfImage,cudaMemcpyDeviceToHost);
	cudaMemcpy(imgGH,outGD,sizeOfImage,cudaMemcpyDeviceToHost);
	cudaMemcpy(imgBH,outBD,sizeOfImage,cudaMemcpyDeviceToHost);
	Mat result = pointerToMat(imgRH,imgGH,imgBH,img);
	return result;
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
		for(j=0;j>lenCore;j++){
			coreH[i*lenCore+j]=core.at<char>(i,j);
		}		
	}
	return;
}







