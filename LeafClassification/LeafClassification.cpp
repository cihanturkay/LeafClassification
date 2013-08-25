// LeafClassification.cpp : Defines the entry point for the console application.
#include "stdAfx.h"
#include <cv.h>
#include<iostream>
#include<stdlib.h>
#include<stdint.h>
#include <cxcore.h>
#include <highgui.h>
#include <iostream>
#include "opencv2/features2d/features2d.hpp"
#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2\core\types_c.h>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "LeafClassifier.h"
#include <ml.h>

using namespace cv;
using namespace std;

bool isBlurry(String path) {
		IplImage* img = cvLoadImage(path.c_str(),0);
		assert( img->width%2 == 0 && img->height%2 == 0);
		IplImage* out = cvCreateImage( cvSize(img->width/2,img->height/2), img->depth, img->nChannels );
		cvPyrDown( img, out );
		cvCanny( out, out,10, 600, 3);

		Mat imgg(out);
		Mat horizontalRow;
		imgg.row(imgg.size().height/2).copyTo(horizontalRow);
		
		uint8_t* pixelPtr = (uint8_t*)horizontalRow.data;		
		int sum = 0;
		for(int j = 0; j < horizontalRow.cols; j ++)
			{				
				sum += pixelPtr[j];
			}
		return sum <= 0;
}

int main(int argc, _TCHAR* argv[])
{

	/*
	LeafClassifier* classifier = new LeafClassifier();
	classifier->Initialize("SVMtrain.txt","BOWtrain.txt" , 1277, true);	

	cout<<endl<<"Test result error rate : "<<classifier->TestClassifier("test.txt",1235)<<endl;
	*/

	/*
	for(int i = 1; i <= 11 ; i++){
	String path = "C:\\Users\\Cihan\\Desktop\\blur\\";
	stringstream ss;
	ss << i;
	path.append(ss.str());
	path.append(".jpg");
	bool result = classifier->isLeafFound(path);
	cout<<i<<" : "<<result<<endl;
	}
	*/
	int falseNum = 0;

	for(int i = 1; i <= 65; i++){
		String path = "C:\\Users\\Cihan\\Desktop\\blur\\";
		stringstream ss;
		ss << i;
		path.append(ss.str());
		path.append(".png");
		bool is = isBlurry(path);
		cout<<" DIR :: "<<path<<" --> "<<is<<endl;
		if(!is)
			falseNum++;
	}

	cout<<falseNum<<endl;
	getchar();
	return 0;
}
