#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <stdlib.h>
#include "cvlH.cu"
using namespace std;
using namespace cv;
Mat cudaCvl(Mat img,Mat core);
Mat cpuCvl(Mat img,Mat core);
char getValue(Mat img,int x,int y,int z,Mat core);
int main(){
	Mat image;
	image = imread("../res/lena.jpg",1);
	Mat image1;
	resize(image,image1,Size(32,32),0,0,CV_INTER_LINEAR);
//	imshow("Display Image",image);
	int lenCore = 5;
	Mat core =(Mat_<char>(lenCore,lenCore)<<1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1);

	Mat image2 = cpuCvl(image1,core);
//	imshow("CPU",image2);

	Mat image3;
	image3 = cudaCvl(image1,core);
//	imshow("CUDA",image3);	

	waitKey(0);
	return 0;
}
char getValue(Mat img,int x,int y,int z,Mat core){
	int lenX = img.cols;
	int lenY = img.rows;
	int lenCore = core.cols;
	if(x-lenCore/2<0|x+lenCore/2>=lenX|y-lenCore/2<0|y+lenCore/2>=lenY){
		return img.at<Vec3b>(x,y)[z];
	}
	int i,j;
	int tmpX,tmpY;
	int sum=0;
	for(i=0;i<lenCore;i++){
		for(j=0;j<lenCore;j++){
			tmpX=x-lenCore/2+i;
			tmpY=y-lenCore/2+j;
			sum+=img.at<Vec3b>(tmpX,tmpY)[z]*core.at<char>(i,j);
		}
	}
	return (char)(sum*1.0/(lenCore*lenCore));
}
Mat cpuCvl(Mat img,Mat core){
	Mat result = img.clone();
	int lenX = img.cols;
	int lenY = img.rows;
	int i,j;
	for(i=0;i<lenX;i++){
		for(j=0;j<lenY;j++){
			result.at<Vec3b>(i,j)[0]=getValue(img,i,j,0,core);
			result.at<Vec3b>(i,j)[1]=getValue(img,i,j,1,core);
			result.at<Vec3b>(i,j)[2]=getValue(img,i,j,2,core);
		}
	}
	return result;
}











