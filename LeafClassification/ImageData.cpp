#include "StdAfx.h"
#include "ImageData.h"


ImageData::ImageData()
{
}

ImageData::ImageData(char* path, int label)
{
	this->path = path;
	this->label = label;
}


ImageData::~ImageData(void)
{
}
