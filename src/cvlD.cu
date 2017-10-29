//device code
#include <stdlib.h>  
#include <cuda_runtime.h> 
#include <stdio.h>  

//设备端代码
__global__ void cvlUnit(const char *imgR,const char *imgG,const char *imgB,const char *core,char *outR,char *outG,char *outB,int lenX,int lenY,int lenCore)  
{  
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int index = x*lenX+y;
	//judgement of index
	if(x<0|x>=lenX|y<0|y>=lenY){
		return;
	}
	//judgement of boundary,return value directly
	if(x-lenCore/2<0|x+lenCore/2>=lenX|y-lenCore/2<0|y+lenCore/2>=lenY){
		outR[index]=imgR[index];
		outG[index]=imgG[index];
		outB[index]=imgB[index];
		return;
	}
	int i,j;
	int tmpX,tmpY;
	int sumR=0,sumG=0,sumB=0;
	for(i=0;i<lenCore;i++){
		for(j=0;j<lenCore;j++){
			tmpX = x-lenCore/2+i;
			tmpY = y-lenCore/2+j;
			sumR += imgR[tmpX*lenX+tmpY]*core[i*lenCore+j];
			sumG += imgG[tmpX*lenX+tmpY]*core[i*lenCore+j];
			sumB += imgB[tmpX*lenX+tmpY]*core[i*lenCore+j];
		}
	}
	outR[index]=(char)(sumR*1.0/(lenCore*lenCore));
	outG[index]=(char)(sumG*1.0/(lenCore*lenCore));
	outB[index]=(char)(sumB*1.0/(lenCore*lenCore));
	return;
}  
