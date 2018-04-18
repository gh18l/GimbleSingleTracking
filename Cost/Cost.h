#ifndef __COST_HPP__
#define __COST_HPP__

#include <iostream>
#include <cstdlib>
#include <chrono>
#include <memory>
#include <opencv2\opencv.hpp>
#include "opencv2/core.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "YOLODetector.hpp"
#include "kcftracker.hpp"

#include "RobustStitcher/FeatureMatch.h"
#include "GenCameraDriver.h"

using namespace std;
using namespace cv;
class Cost
{
public:
	Cost();
	~Cost();
private:
	std::shared_ptr<YOLODetector> detector = std::make_shared<YOLODetector>();
	std::shared_ptr<YOLODetector> detector_face = std::make_shared<YOLODetector>();
	std::vector<cv::Rect> img_rect;
	KCFTracker KCF_tracker;
	///////the number of blocks
	int rows_b = 4;
	int cols_b = 4;
	///////the number of input net
	int set_b = 4;

	///////set the face blocks
	double width_f = 450.0;
	double height_f = 400.0;

	//////when the tracking happens, up/downsampling the image to reduce runtime
	float sam_scale = 1.0f;
	cv::Rect crop_roi;

	/////isfind
	int isfind_time = 0;
	std::vector<cv::Point> isfind_vec;


public:
	cv::Point people_point;
	float people_constant = 1;
	float people_flow_gain = 1;

	//////video flow////
	cv::Ptr<cv::BackgroundSubtractor> bg_model = createBackgroundSubtractorMOG2().dynamicCast<cv::BackgroundSubtractor>();
	bool smoothMask = 0;   
	bool update_bg_model = true;
	///cost///
	cv::Mat fre;
	int fre_tre = 10;

	std::vector<bbox_t> detect_vec;
	bbox_t current_vec;
	////tracking
	bool istracking = 0;
	cv::Rect tracking_roi;
	////tracking block
	float scale;
	float finalwidth, finalheight;

	////show
	cv::Mat show_opencv;
	cv::Mat show_opencv2;
	int thread_flag = 0;

	int isfind_max;
	bool find_face = 0;

	int Thread_end = 0;
private:
	bool isfind();
	std::vector<bbox_t> detection(cv::Mat img);    //only detect the people
public:
	int init_people_detection(cv::Mat img);
	int init_face_detection();
	int people_detection(cv::Mat& img);
	cv::Mat cal_dis(cv::Mat flow_uv0, cv::Mat flow_uv1);
	int flow(cv::Mat src1, cv::Mat src2, cv::Point &point);
	int video_updateflow(cv::Mat src, cv::Point& point);
	int video_initflow(cv::Mat src);
	int video_preflow(cv::Mat src);
	int SetBlock(cv::Mat img);
	cv::Mat SetFaceBlock(cv::Mat ref_people, cv::Mat local);
	int GetSum(cv::Mat input);
	std::vector<int> choose_maxs(std::vector<int> sum);
	int find_min(std::vector<int> index, std::vector<int>sum);
	int video_updatedetection(cv::Mat src, cv::Point src_point, cv::Point& dst_point);

	int tracking_init(cv::Mat frame);
	bool tracking(cv::Mat frame);
	int Thtracking();
	void startTh();
	bool iscontain(bbox_t roi);

	double people_match(cv::Mat img1, cv::Mat img2);
	int face_detection(cv::Mat local);



private:
	
public:
	cv::Mat ref_people;
	std::vector<cv::Mat> current_show;
};

#endif
