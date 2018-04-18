/**
@brief class for Stitcher
@author: Shane Yuan
@date: Dec 11, 2017
*/

#ifndef __TINY_GIGA_STITCHER_HPP__
#define __TINY_GIGA_STITCHER_HPP__


#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include <opencv2/optflow.hpp>

#include "SysUtil.hpp"
#include "ASAP.h"
#include "FeatureMatch.h"
#include "DeepFlow.h"
#include "Serial.h"
#define PM 6200
#define YM 3900



class TinyStitcher {
public:    //private:
	// input variable
	cv::Mat refImg;
	cv::Mat localImg;
	float scale;
	std::string modelname;
	std::shared_ptr<FeatureMatch> const featureMatchPtr = std::make_shared<FeatureMatch>();
	// first iteration
	cv::Size sizeBlock;
	cv::Mat refBlk;
	cv::Rect refRect;
	cv::Ptr<cv::ximgproc::StructuredEdgeDetection> ptr;
	// second iteration
	std::vector<cv::Point2f> refPts_seconditer;
	std::vector<cv::Point2f> localPts_seconditer;
	cv::Mat globalH;
	// third iteration
	std::vector<cv::Point2f> refPts_thirditer;
	std::vector<cv::Point2f> localPts_thirditer;
	std::shared_ptr<ASAP> const asapPtr_thirditer = std::make_shared<ASAP>();
	cv::Mat warpImg_thirditer;
	// fourth iteration
	std::vector<cv::Point2f> refPts_fourthiter;
	std::vector<cv::Point2f> localPts_fourthiter;
	std::shared_ptr<ASAP> const asapPtr_fourthiter = std::make_shared<ASAP>();
	std::shared_ptr<cv::optflow::DeepFlow> deepflowPtr 
		= std::make_shared<cv::optflow::DeepFlow>();
	cv::Mat warpImg_fourthiter;
	float dX = 40.0, X_MIN = 0, dY = 30.0, Y_MIN = 0;
	int Row = 11;
	int Col = 9;
/////////////////////////////new//////////////////////////
public:
	cv::Point current_point;
	cv::Point2f current_pulse;
	cv::Mat mask;


public:
	/**
	@brief apply SED detector on input image
	@param cv::Mat img: input image
	@param float scale: resize scale (apply SED detector on original size image is
			too time consuming)
	@return cv::Mat: returned edge image
	*/
	cv::Mat SEDDetector(cv::Mat img, float scale);

	/**
	@brief color correction (gray image only)
	@param cv::Mat & srcImg: input/output src gray image
	@param cv::Mat dstImg: input dst gray image
	@return int
	*/
	int colorCorrect(cv::Mat & srcImg, cv::Mat dstImg);
	int TinyStitcher::colorCorrectRGB(cv::Mat & srcImg, cv::Mat dstImg);
	/**
	@brief re-sample feature points from optical flow fields
	@param cv::Mat refinedflowfield: input refined flow fields
	@param cv::Mat preflowfield: input prewarped flow fields
	@param std::vector<cv::Point2f> & refPts_fourthiter: output reference feature points
	@param std::vector<cv::Point2f> & localPts_fourthiter: output local feature points
	@param int meshrows: rows of mesh
	@param int meshcols: cols of mesh
	@return int
	*/
	int resampleFeaturePoints(cv::Mat refinedflowfield, cv::Mat preflowfield,
		std::vector<cv::Point2f> & refPts_fourthiter,
		std::vector<cv::Point2f> & localPts_fourthiter, int meshrows, int meshcols);

public:
	TinyStitcher();
	~TinyStitcher();

	/**
	@brief init tiny giga
	@param cv::Mat refImg: input reference image
	@param cv::Mat localImg: input local-view image
	@param std::string modelname: modelname used for structure edge detector
	@param float scale: input scale
	@return int
	*/
	int init(cv::Mat refImg, cv::Mat localImg, std::string modelname, float scale);

	/**
	@brief first iteration, find reference block
	@return int
	*/
	int firstIteration();

	/**
	@brief second iteration, find global homography 
	@return int
	*/
	int secondIteration();

	/**
	@brief third iteration, find local homography matrices
	@return int
	*/
	int thirdIteration();

	/**
	@brief fourth iteration, use optical flow to refine local homography matrices
	@return int
	*/
	int fourthIteration();

	/**
	@brief get final warped image
	@return cv::Mat warpImg: finally warped image
	*/
	cv::Mat getFinalWarpImg();

	/**
	@brief get reference block image
	@return cv::Mat warpImg: reference block image
	*/
	cv::Mat getRefBlkImg();

	/****************************new******************************/
public:
	int calibration_record(float start_position, float end_position);
	cv::Mat merge_copy(cv::Mat src);
	int test_100(cv::Point2f src, CSerial serial);
	int move(cv::Point& dst);
	int detect_move(cv::Point dst);
	int find_position(cv::Mat refImg, cv::Mat localImg, cv::Point &out_point);
	int init_stitcher();
	int gimble_find_position(cv::Mat refImg, cv::Mat localImg, cv::Point ref_point, float region_mul,cv::Point &out_point);
	bool isInside(cv::Point2f pt, cv::Rect rect);
	/********************************************************************/
	/*                     visualization functions                      */
	/********************************************************************/
	/**
	@brief visualize matching points
	@param cv::Mat img1: first image
	@param cv::Mat img2: second image
	@param std::vector<cv::Point2f> pt1: matching points of the first image
	@param std::vector<cv::Point2f> pt2: matching points of the second image
	@param int direction, 0: horizontal, 1: vertical
	*/
	static cv::Mat visualMatchingPts(cv::Mat img1, cv::Mat img2,
		std::vector<cv::Point2f> pt1, std::vector<cv::Point2f> pt2, int direction = 0);
	//int TinyStitcher::global2pano(cv::Mat global, cv::Mat pano);

};

#endif