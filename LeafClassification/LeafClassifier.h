#pragma once

#include "ImageData.h"
#include <cv.h>
#include <cxcore.h>
#include <ml.h>
#include<iostream>
#include <cv.h>
#include <cxcore.h>
#include <ml.h>
#include <fstream>
#include <string>
#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>


using namespace cv;

class LeafClassifier
{
	public:
	CvSVM SVM;
	int trainSize;
	ImageData* BowtrainSetImageInfo;
	ImageData* SVMtrainSetImageInfo;
	CvTermCriteria criteria;
	Ptr<BOWImgDescriptorExtractor> bowide;
	Ptr<FeatureDetector> detector;
	Ptr<DescriptorExtractor> extractor;
	Ptr<DescriptorMatcher> matcher;
	int dictionarySize;

	LeafClassifier(void);
	void Initialize(char* SVMfilePath, char* BOWfilePath, int trainSize, bool useSaveModel);
	double TestClassifier(char * path, int size);
	~LeafClassifier(void);
	void SaveModel(char* fileName);
	void ExtractFeatures(bool useSaveModel);
	void ReadTrainDataFromFile(char* filePath, int size, ImageData* imagesData);
	bool isLeafFound(char* imagePath);
	bool readVocabulary( const string& filename, Mat& vocabulary );
	bool writeVocabulary( const string& filename, const Mat& vocabulary );

};
