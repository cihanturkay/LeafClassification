#pragma once

#include "cv.h";

class ImageData
{
public:
	char* path;
	int label;
	
	ImageData(void);
	ImageData(char* path, int label);
	~ImageData(void);
};

