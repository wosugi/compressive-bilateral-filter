#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#define CV_VERSION_NUMBER CVAUX_STR(CV_MAJOR_VERSION) CVAUX_STR(CV_MINOR_VERSION) CVAUX_STR(CV_SUBMINOR_VERSION)

#ifdef _DEBUG
#pragma comment(lib, "opencv_core"CV_VERSION_NUMBER"d.lib")
#pragma comment(lib, "opencv_highgui"CV_VERSION_NUMBER"d.lib")
#pragma comment(lib, "opencv_imgproc"CV_VERSION_NUMBER"d.lib")
//#pragma comment(lib, "opencv_contrib"CV_VERSION_NUMBER"d.lib")
#else
#pragma comment(lib, "opencv_core"CV_VERSION_NUMBER".lib")
#pragma comment(lib, "opencv_highgui"CV_VERSION_NUMBER".lib")
#pragma comment(lib, "opencv_imgproc"CV_VERSION_NUMBER".lib")
//#pragma comment(lib, "opencv_contrib"CV_VERSION_NUMBER".lib")
#endif

int main(int argc,char** argv)
{
	if(argc!=2)
	{
		std::cerr<<"[Usage] CompressiveBilateralFilter (ImagePath)"<<std::endl;
		return 1;
	}

	const std::string& path(argv[1]);
	cv::Mat_<cv::Vec3b> image=cv::imread(path);
	cv::imshow("input image",image);
	cv::waitKey();
	return 0;
}
