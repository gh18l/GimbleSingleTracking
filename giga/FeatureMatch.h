/**
@brief feature matching
@author: Shane Yuan
@date: Dec 11, 2017
*/

#ifndef __TINY_GIGA_FEATURE_MATCH_H__
#define __TINY_GIGA_FEATURE_MATCH_H__

#include <iostream>
#include <fstream>
#ifdef WIN32
#include <Windows.h>
#endif
#include <direct.h>
#include <chrono>
#include <memory>
#include <thread>

#include <opencv2/opencv.hpp>


class FeatureMatch {
private:
	cv::Size patchSize;
	cv::Size searchSize;

public:

private:
	/**
	@brief check if a point is in a rect
	@param cv::Point2f pt: input point
	@param cv::Rect rect: input rect
	@return bool: if inside rectagnle
	*/
	bool isInside(cv::Point2f pt, cv::Rect rect);

	/**
	@brief detect ket points (using goodfeature2track algorithm)
	@param cv::Mat img: input image
	@param std::vector<cv::Point2f> & keypts: output key points
	@param float scale = 1.0f: scale used for feature matching (directly operating on 
		original scale is slow and time consuming)
	@return int
	*/
	int detectKeyPts(cv::Mat img, std::vector<cv::Point2f> & keypts, float scale = 1.0f);

	/**
	@brief calculate feature points between images
	@param cv::Mat refBlk: reference image
	@param cv::Mat localImg: local image
	@param std::vector<cv::Point2f> refPts: reference points
	@param std::vector<cv::Point2f> localPts: localview points
	@return int
	*/
	int match(cv::Mat refBlk, cv::Mat localImg, std::vector<cv::Point2f> refPts,
		std::vector<cv::Point2f> localPts, std::vector<cv::Point2f> & outRefPts,
		std::vector<cv::Point2f> & outLocalPts);
	
	/**
	@brief global ransac to discard outliers
	@param std::vector<cv::Point2f> & refPts: input/output reference points
	@param std::vector<cv::Point2f> & localPts: input/output localview points
	@param float threshold: threshold used in ransac
	@return int
	*/
	int gloablRansac(std::vector<cv::Point2f> & refPts, std::vector<cv::Point2f> & localPts, 
		float threshold);

	/**
	@brief local ransac to discard outliers
	@param std::vector<cv::Point2f> & refPts: input/output reference points
	@param std::vector<cv::Point2f> & localPts: input/output localview points
	@param int width: width of image
	@param int height: height of image
	@param int rows: rows of local ransac
	@param int cols: cols of local ransac
	@param float threshold: threshold used in ransac
	@return int
	*/
	int localRansac(std::vector<cv::Point2f> & refPts, std::vector<cv::Point2f> & localPts,
		int width, int height, int rows, int cols, float threshold);

public:
	FeatureMatch();
	~FeatureMatch();

	/**
	@brief set size
	@param cv::Size patchSize: size of patch (template)
	@param cv::Size searchSize: size of searching range
	@return int
	*/
	int setSize(cv::Size patchSize, cv::Size searchSize);

	/**
	@brief build correspondecne for two input images (no global homography matrix provided)
	@param cv::Mat refBlk: input low quality reference image
	@param cv::Mat localImg: input high quality localview image
	@param float scale: scale used for feature matching (directly operating on 
		original scale is slow and time consuming)
	@param std::vector<cv::Point2f> & outRefPts: output reference feature points
	@param std::vector<cv::Point2f> & outLocalPts: output local feature points
	@return int
	*/
	int buildCorrespondence(cv::Mat refBlk, cv::Mat localImg, float scale,
		std::vector<cv::Point2f> & outRefPts, std::vector<cv::Point2f> & outLocalPts);

	/**
	@brief build correspondecne for two input images (global homography matrix provided)
	@param cv::Mat refBlk: input low quality reference image
	@param cv::Mat localImg: input high quality localview image
	@param float scale: scale used for feature matching (directly operating on
	original scale is slow and time consuming)
	@param cv::Mat globalH: gloabl homography matrix
	@param std::vector<cv::Point2f> & outRefPts: output reference feature points
	@param std::vector<cv::Point2f> & outLocalPts: output local feature points
	@return int
	*/
	int buildCorrespondence(cv::Mat refBlk, cv::Mat localImg, float scale, cv::Mat globalH,
		std::vector<cv::Point2f> & outRefPts, std::vector<cv::Point2f> & outLocalPts);

};


#endif