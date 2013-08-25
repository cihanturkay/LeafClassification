#include "StdAfx.h"
#include <cv.h>
#include "LeafClassifier.h"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2\nonfree\features2d.hpp>
#include <opencv2\nonfree\nonfree.hpp>
#include <opencv2\legacy\legacy.hpp>
using namespace std;
using namespace cv;

LeafClassifier::LeafClassifier(void)
{

	dictionarySize = 1000;
	this->extractor = Ptr<DescriptorExtractor>(new SurfDescriptorExtractor()); 
	this->matcher = Ptr<DescriptorMatcher>(new BruteForceMatcher<L2<float>>());
	this->bowide = Ptr<BOWImgDescriptorExtractor>(new BOWImgDescriptorExtractor(extractor, matcher));
}

void LeafClassifier::Initialize(char* SVMpath, char *BOWpath, int trainSize, bool useSaveModel)
{
	this->trainSize = trainSize;
	this->BowtrainSetImageInfo = new ImageData[this->trainSize];
	this->SVMtrainSetImageInfo = new ImageData[this->trainSize];
	//this->trainSetImageInfo = new ImageData[this->trainSize];
	//this->ReadTrainDataFromFile(path, this->trainSize, this->trainSetImageInfo);
	if(!useSaveModel){
		this->ReadTrainDataFromFile(BOWpath, this->trainSize, this->BowtrainSetImageInfo);
		this->ReadTrainDataFromFile(SVMpath, this->trainSize, this->SVMtrainSetImageInfo);
	}
	this->ExtractFeatures(useSaveModel);
}

double LeafClassifier::TestClassifier(char * path,int size)
{
	ImageData *testSetImageInfo = new ImageData[size];
	this->ReadTrainDataFromFile(path, size, testSetImageInfo);
	double response = 0;

	for(int i=0; i< size; i++)
	{		
		bool isLeaf = isLeafFound((testSetImageInfo + i)->path);
		if(isLeaf){
			response++;
		}

		cout<<"isLeaf found for "<<(testSetImageInfo + i)->path<<" ::: "<<isLeaf<<endl;
	}
	cout<<endl<<"response : "<<response;
	cout<<endl<<"size : "<<size;
	return (double)((double)size-response) / (double)size;
}

bool LeafClassifier::isLeafFound(char* imagePath)
{
	Mat image = imread(imagePath, CV_LOAD_IMAGE_UNCHANGED);
	vector<KeyPoint> keypoints;
	Mat bowDescriptor;

	while(image.size().height > 640 || image.size().width > 640){
				//cout<<"image width : "<<img.size().width<<" height : "<< img.size().height<<endl;
				resize(image, image, Size(), 0.5, 0.5, CV_INTER_AREA);
			}

	detector->detect(image, keypoints);
	bowide->compute(image, keypoints, bowDescriptor);

	float response = this->SVM.predict(bowDescriptor);
	image.release();
	return (response == 1.0);
}

void LeafClassifier::ExtractFeatures(bool useModel)//cv::Mat& image
{
	TermCriteria termCriteria(CV_TERMCRIT_EPS, 100, 0.001);
	int retries = 1;
	int flags = KMEANS_PP_CENTERS;
	//detecting keypoints

	detector = new SurfFeatureDetector(500);
	vector<KeyPoint> keypoints;	
	//computing descriptors

	Mat descriptors;
	Mat vocabulary;
	Mat training_descriptors(1,extractor->descriptorSize(),extractor->descriptorType());
	Mat img;

	if(useModel) 
	{

		readVocabulary("BOWModel_v_3",vocabulary);
		cout<<"vocabulary read : "<<vocabulary.depth()<<endl;
	}else{

		for(int i= 0; i< this->trainSize; i++) 
		{
			cout<<"training boww"<<(this->BowtrainSetImageInfo+i)->path<<endl;
			img = imread((this->BowtrainSetImageInfo+i)->path, CV_LOAD_IMAGE_UNCHANGED);
				while(img.size().height > 640 || img.size().width > 640){
				//cout<<"image width : "<<img.size().width<<" height : "<< img.size().height<<endl;
				resize(img, img, Size(), 0.5, 0.5, CV_INTER_AREA);
			}

			detector->detect(img, keypoints);
			cout<<"keypoint : "<<keypoints.size()<<endl;
			extractor->compute(img, keypoints, descriptors);
			cout<<"descriptor : "<<descriptors.depth()<<endl;
			training_descriptors.push_back(descriptors);
			img.release();
		}
		cout<<"saving boww"<<endl;
		BOWKMeansTrainer bowTrainer(dictionarySize, termCriteria, retries, flags);
		bowTrainer.add(training_descriptors);
		vocabulary = bowTrainer.cluster();
		cout<<"--vocabulary saving---"<<endl;
		writeVocabulary("BOWModel_v_4",vocabulary);
		cout<<"--vocabulary saved ----"<<endl;

	}	

	bowide->setVocabulary(vocabulary);
	cout<<"saved boww"<<endl;
	cout << "------- train SVMs ---------\n";

	if(useModel) 
	{
		cout << "------- loading trained SVMs ---------\n";
		this->SVM.load("SVMModel_v_3.txt", 0);
	}
	else
	{
		Mat labels(0, 1, CV_32FC1);
		Mat trainingData(0, dictionarySize, CV_32FC1);
		Mat img2;
		Mat bowDescriptor1;
		vector<KeyPoint> keypoint1;

		for(int i= 0; i< this->trainSize; i++) 
		{
			img2 = imread((this->SVMtrainSetImageInfo+i)->path, CV_LOAD_IMAGE_UNCHANGED);
				while(img.size().height > 640 || img.size().width > 640){
				//cout<<"image width : "<<img.size().width<<" height : "<< img.size().height<<endl;
				resize(img, img, Size(), 0.5, 0.5, CV_INTER_AREA);
			}
			detector->detect(img2, keypoint1);
			bowide->compute(img2, keypoint1, bowDescriptor1);
			if(!bowDescriptor1.empty()){				
			trainingData.push_back(bowDescriptor1);
			labels.push_back((float)(this->SVMtrainSetImageInfo+i)->label);
			//cout<<"training svm"<<(this->SVMtrainSetImageInfo+i)->path<<endl;
			}
			img2.release();
		}
		cout<<"end of svm images"<<endl;

		CvSVMParams params;
		params.kernel_type=CvSVM::RBF;
		params.svm_type=CvSVM::C_SVC;
		params.gamma=0.50625000000000009;
		params.C=312.50000000000000;
		params.term_crit=cvTermCriteria(CV_TERMCRIT_ITER,100,0.000001);

		SVM.train_auto(trainingData,labels,Mat(), Mat(), params, 10);
		params = SVM.get_params();
		printf( "\nUsing optimal parameters degree %f, gamma %f, ceof0 %f\n\t C %f, nu %f, p %f\n Training ..",
                params.degree, params.gamma, params.coef0, params.C, params.nu, params.p);
		
		//this->SVM.train(trainingData,labels,cv::Mat(),cv::Mat(),params);
		cout<<"saving SVM"<<endl;
		this->SVM.save("SVMModel_v_4.txt");
	}
}

void LeafClassifier::SaveModel(char* fileName)
{
	this->SVM.save(fileName); 
}

void LeafClassifier::ReadTrainDataFromFile(char* fileName, int size, ImageData* imagesData)
{
	int i= 0;
	string line;
	ifstream file(fileName);

	if (file.is_open())
	{
		while(i != size)
		{
			getline (file,line);

			char *charOfLine= new char[line.size()+1];
			charOfLine[line.size()]= 0;
			memcpy(charOfLine, line.c_str(), line.size());
			imagesData[i].path =  strtok(charOfLine, " ");

			if(line[line.size()-1] == '1')
			{
				imagesData[i].label = 1;
			}
			else 
			{
				imagesData[i].label = 0;//"line;
			}

			cout << line << endl;
			i++;
		}

		file.close();
	}
	else 
	{
		cout << "Unable to open file"; 
		exit(1);
	}
}

bool LeafClassifier::readVocabulary( const string& filename, Mat& vocabulary )
{
	cout << "Reading vocabulary...";
	FileStorage fs( filename, FileStorage::READ );
	if( fs.isOpened() )
	{
		fs["vocabulary"] >> vocabulary;
		cout << "done" << endl;
		return true;
	}
	return false;
}

bool LeafClassifier::writeVocabulary( const string& filename, const Mat& vocabulary )
{
	cout << "Saving vocabulary..." << endl;
	FileStorage fs( filename, FileStorage::WRITE );
	if( fs.isOpened() )
	{
		fs << "vocabulary" << vocabulary;
		return true;
	}
	return false;
}

LeafClassifier::~LeafClassifier(void)
{

}
