#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/videoio.hpp>

#include "kcftracker.hpp"

//#include <dirent.h>

using namespace std;
using namespace cv;

int main(int argc, char* argv[]){

	if (argc > 5) return -1;

	bool HOG = true;
	bool FIXEDWINDOW = false;
	bool MULTISCALE = false;
	bool SILENT = false;
	bool LAB = false;

	for(int i = 0; i < argc; i++){
		if ( strcmp (argv[i], "hog") == 0 )
			HOG = true;
		if ( strcmp (argv[i], "fixed_window") == 0 )
			FIXEDWINDOW = true;
		if ( strcmp (argv[i], "singlescale") == 0 )
			MULTISCALE = false;
		if ( strcmp (argv[i], "show") == 0 )
			SILENT = false;
		if ( strcmp (argv[i], "lab") == 0 ){
			LAB = true;
			HOG = true;
		}
		if ( strcmp (argv[i], "gray") == 0 )
			HOG = false;
	}
	
	// Create KCFTracker object
	KCFTracker tracker(HOG, FIXEDWINDOW, MULTISCALE, LAB);
	std::string video = "E:/code/YOLO/darknet-master/Sln/ref.avi";
	VideoCapture cap(video);

	// Frame readed

	// Tracker results
	Rect result;
	cv::Mat temp;
	Mat frame;
	


	// Write Results
	ofstream resultsFile;
	string resultsPath = "output.txt";
	resultsFile.open(resultsPath);
	for (int i = 0; i < 200; i++)
	{
		cap >> frame;
	}
	// Frame counter
	int nFrames = 0;


	while (1){

		// Read each frame from the list
		cap >> frame;
		// First frame, give the groundtruth to the tracker
		if (nFrames == 0) {
			Rect2d region = selectROI("tracker", frame, true, false);
			cv::resize(frame, frame, Size(frame.cols * 4, frame.rows * 4));
			region.x *= 4;
			region.y *= 4;
			region.width *= 4;
			region.height *= 4;
			tracker.init(region, frame);
			rectangle(frame, region, Scalar( 255, 0, 0 ), 5, 8 );
		}
		// Update
		else{
			cv::resize(frame, frame, Size(frame.cols * 4, frame.rows * 4));
			result = tracker.update(frame);
			rectangle( frame, Point(result.x, result.y), Point( result.x+result.width, result.y+result.height), Scalar( 255, 0, 0 ), 5, 8 );
		}

		nFrames++;

		if (!SILENT){
			cv::resize(frame, temp, Size(800, 600));
			imshow("Image", temp);
			waitKey(1);
		}
	}


}
