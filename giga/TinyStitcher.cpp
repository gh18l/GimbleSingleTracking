/**
@brief class for Stitcher
@author: Shane Yuan
@date: Dec 11, 2017
*/

#include "TinyStitcher.h"
#include <opencv2/ximgproc.hpp>
#include<opencv2\opencv.hpp>
#include <time.h>
#include "Serial.h"
#include "GenCameraDriver.h"

#define _DEBUG_TINY_STITCHER
TinyStitcher::TinyStitcher() {}
TinyStitcher::~TinyStitcher() {}

clock_t start, end;
double dur;

extern float delta_pixel;
/**
@brief init tiny giga
@param cv::Mat refImg: input reference image
@param cv::Mat localImg: input local-view image
@param std::string modelname: modelname used for structure edge detector
@param float scale: input scale
@return int
*/

bool TinyStitcher::isInside(cv::Point2f pt, cv::Rect rect) {
	if (rect.contains(pt))
		return true;
	else return false;
}

int TinyStitcher::init(cv::Mat refImg, cv::Mat localImg,
	std::string modelname, float scale) {
	this->refImg = refImg;
	this->localImg = localImg;
	this->modelname = modelname;
	this->scale = scale;
	ptr = cv::ximgproc::createStructuredEdgeDetection(modelname);
	return 0;
}

/**
@brief apply SED detector on input image
@param cv::Mat img: input image
@param float scale: resize scale (apply SED detector on original size image is
too time consuming)
@return cv::Mat: returned edge image
*/
cv::Mat TinyStitcher::SEDDetector(cv::Mat img, float scale) {
	cv::Mat edgeImg;
	cv::Size size_large = img.size();
	cv::Size size_small = cv::Size(size_large.width * scale, size_large.height * scale);
	cv::resize(img, img, size_small);
	img.convertTo(img, cv::DataType<float>::type, 1 / 255.0);
	ptr->detectEdges(img, edgeImg);
	edgeImg = edgeImg * 255;
	cv::resize(edgeImg, edgeImg, size_large);
	edgeImg.convertTo(edgeImg, CV_8U);
	return edgeImg;
}

/**
@brief color correction (gray image only)
@param cv::Mat & srcImg: input/output src gray image
@param cv::Mat dstImg: input dst gray image
@return int
*/
int TinyStitcher::colorCorrect(cv::Mat & srcImg, cv::Mat dstImg) {
	cv::Scalar meanSrc, stdSrc, meanDst, stdDst;
	cv::meanStdDev(srcImg, meanSrc, stdSrc);     
	cv::meanStdDev(dstImg, meanDst, stdDst);

	srcImg.convertTo(srcImg, -1, stdDst.val[0] / stdSrc.val[0], 
		meanDst.val[0] - stdDst.val[0] / stdSrc.val[0] * meanSrc.val[0]);
	return 0;
}

int TinyStitcher::colorCorrectRGB(cv::Mat & srcImg, cv::Mat dstImg) 
{
	cv::Scalar meanSrc, stdSrc, meanDst, stdDst;
	cv::meanStdDev(srcImg, meanSrc, stdSrc);
	cv::meanStdDev(dstImg, meanDst, stdDst);
	std::vector<cv::Mat> channel;
	cv::split(srcImg, channel);
	for (int i = 0; i < 3; i++)
	{
		channel[i].convertTo(channel[i], -1, stdDst.val[i] / stdSrc.val[i],
			meanDst.val[i] - stdDst.val[i] / stdSrc.val[i] * meanSrc.val[i]);
	}
	cv::merge(channel, srcImg);
	
	return 0;
}

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
int TinyStitcher::resampleFeaturePoints(cv::Mat refinedflowfield, cv::Mat preflowfield,
	std::vector<cv::Point2f> & refPts_fourthiter,
	std::vector<cv::Point2f> & localPts_fourthiter,
	int meshrows, int meshcols) {
	// calculate quad size
	float quadWidth = static_cast<float>(refinedflowfield.cols) / static_cast<float>(meshcols);
	float quadHeight = static_cast<float>(refinedflowfield.rows) / static_cast<float>(meshrows);
	// init matching points vectors
	std::vector<cv::Point2f> refPts_final;
	std::vector<cv::Point2f> localPts_final;
	// add matching points on key point positions
	for (size_t i = 0; i < refPts_fourthiter.size(); i++) {
		cv::Point2f p = refPts_fourthiter[i];
		cv::Point2f initflowVal = preflowfield.at<cv::Point2f>(p.y, p.x);
		cv::Point2f refinedflowVal = refinedflowfield.at<cv::Point2f>(p.y, p.x);
		if (cv::norm(initflowVal - refinedflowVal) > 400)
			continue;
		cv::Point2f p1 = refinedflowVal;
		localPts_final.push_back(p1);
		refPts_final.push_back(p);
	}
	// add matching points on quad center
	std::vector<cv::Point2f> borderPoints;
	borderPoints.push_back(cv::Point2f(0.5, 0.5));
	borderPoints.push_back(cv::Point2f(0.25, 0.25));
	borderPoints.push_back(cv::Point2f(0.75, 0.25));
	borderPoints.push_back(cv::Point2f(0.75, 0.75));
	borderPoints.push_back(cv::Point2f(0.75, 0.75));
	for (size_t i = 0; i < meshrows; i++) {
		for (size_t j = 0; j < meshcols; j++) {
			int pointNum = 1;
			if (i == 0 || j == 0 || i == meshrows - 1 || j == meshcols - 1)
				pointNum = 5;
			for (size_t k = 0; k < pointNum; k++) {
				cv::Point2f p = cv::Point2f(quadWidth * (j + borderPoints[k].x), 
					quadHeight * (i + borderPoints[k].y));
				cv::Point2f initflowVal = preflowfield.at<cv::Point2f>(p.y, p.x);
				cv::Point2f refinedflowVal = refinedflowfield.at<cv::Point2f>(p.y, p.x);
				if (cv::norm(initflowVal - refinedflowVal) > 400)
					continue;
				cv::Point2f p1 = refinedflowVal;
				localPts_final.push_back(p1);
				refPts_final.push_back(p);
			}
		}
	}
	refPts_fourthiter = refPts_final;
	localPts_fourthiter = localPts_final;
	return 0;
}

/**
@brief first iteration, find reference block
@return int
*/
int TinyStitcher::firstIteration() {
	SysUtil::infoOutput("First iteration ...\n");
	// calculate edge map
	cv::Mat refEdge = SEDDetector(refImg, 0.25);
	cv::Mat localEdge = SEDDetector(localImg, 0.25);
	// resize localview image
	cv::Mat templ, templEdge;
	sizeBlock = cv::Size(localImg.cols * scale, localImg.rows * scale);
	cv::resize(localImg, templ, sizeBlock);
	cv::resize(localEdge, templEdge, sizeBlock);

	cv::Mat result, resultEdge;
	cv::matchTemplate(refImg, templ, result, cv::TM_CCOEFF_NORMED);
	cv::matchTemplate(refEdge, templEdge, resultEdge, cv::TM_CCOEFF_NORMED);
	result = result.mul(resultEdge);

	cv::Point maxLoc;
	cv::minMaxLoc(result, NULL, NULL, NULL, &maxLoc);
	refRect = cv::Rect(maxLoc.x, maxLoc.y, sizeBlock.width, sizeBlock.height);
	refImg(refRect).copyTo(refBlk);
	/////
	cv::resize(localImg, localImg, refBlk.size());
	cv::Mat temp;
	localImg.copyTo(temp);
	refBlk.copyTo(localImg);
	temp.copyTo(refBlk);

	//cv::resize(refBlk, refBlk, localImg.size());

	//////////////


	return 0;
}


/**
@brief second iteration, find global homography
@return int
*/
int TinyStitcher::secondIteration() {
	SysUtil::infoOutput("Second iteration ...\n");
	featureMatchPtr->setSize(cv::Size(256, 256), cv::Size(512, 512));
	featureMatchPtr->buildCorrespondence(refBlk, localImg, 1.0f, refPts_seconditer, localPts_seconditer);    //0.5f
#ifdef _DEBUG_TINY_STITCHER
	cv::Mat visual = TinyStitcher::visualMatchingPts(refBlk, localImg, refPts_seconditer,
		localPts_seconditer, 0);
#endif
	globalH = cv::findHomography(localPts_seconditer, refPts_seconditer, CV_RANSAC, 3.0f);
	globalH.convertTo(globalH, CV_32F);
	return 0;
}

/**
@brief third iteration, find local homography matrices
@return int
*/
int TinyStitcher::thirdIteration() {
	SysUtil::infoOutput("Third iteration ...\n");
	// build correspondence
	start = clock();
	featureMatchPtr->buildCorrespondence(refBlk, localImg, 1.0f, globalH,
		refPts_thirditer, localPts_thirditer);      ////////////////////////////////////
	end = clock();
	dur = (double)(end - start);
	std::cout << "buildCorrespondence Use Time" << (dur / CLOCKS_PER_SEC) << end;
#ifdef _DEBUG_TINY_STITCHER
	cv::Mat visual = TinyStitcher::visualMatchingPts(refBlk, localImg, refPts_seconditer,
		localPts_seconditer, 0);
#endif
	// apply asap to get deformation mesh
	int width = refBlk.cols;
	int height = refBlk.rows;
	int meshrows = 8;
	int meshcols = 8;  //8*8的格子
	float quadWidth = static_cast<float>(width) / static_cast<float>(meshcols);    //每个格子的宽高
	float quadHeight = static_cast<float>(height) / static_cast<float>(meshrows);
	float smoothWeight = 0.5;
	start = clock();
	asapPtr_thirditer->setMesh(height, width, quadHeight, quadWidth, smoothWeight);
	asapPtr_thirditer->setControlPoints(localPts_thirditer, refPts_thirditer);
	asapPtr_thirditer->solve();
	asapPtr_thirditer->calcFlowField();
	cv::Mat flow = asapPtr_thirditer->getFlowfield();
	cv::remap(localImg, warpImg_thirditer, flow, cv::Mat(), cv::INTER_LINEAR);
	end = clock();
	dur = (double)(end - start);
	std::cout << "buildCorrespondence Use Time" << (dur / CLOCKS_PER_SEC) << end;
	return 0;
}///////////基于特征点通过mesh-based得到source mesh的target mesh不规则角点，根据四个角点求一个透视变换矩阵，再用这个矩阵得到
////////////源矩阵中warping过去的目的点坐标。

/**
@brief fourth iteration, use optical flow to refine local homography matrices
@return int
*/
int TinyStitcher::fourthIteration() {
	SysUtil::infoOutput("Fourth iteration ...\n");
	// optical flow refine
	float flowCalcScale = 0.25;
	cv::Mat refineflow;
	cv::Mat localImgGray, refImgGray;
	cv::Size newsize(static_cast<float>(refBlk.cols) * flowCalcScale,
		static_cast<float>(refBlk.rows) * flowCalcScale);
	cv::resize(warpImg_thirditer, localImgGray, newsize);
	cv::resize(refBlk, refImgGray, newsize);
	cv::cvtColor(refImgGray, refImgGray, CV_BGR2GRAY);
	cv::cvtColor(localImgGray, localImgGray, CV_BGR2GRAY);
	// color correction
	colorCorrect(localImgGray, refImgGray);

	// calculate optical flow
	cv::Mat updateflow;
	cv::Ptr<cv::optflow::DeepFlow> deepFlow = cv::optflow::createDeepFlow();
	deepFlow->calc(refImgGray, localImgGray, updateflow);
	cv::resize(updateflow, updateflow, refBlk.size());
	updateflow = updateflow / flowCalcScale;
	for (int i = 0; i < updateflow.rows; i++) {
		for (int j = 0; j < updateflow.cols; j++) {
			cv::Point2f _flow = updateflow.at<cv::Point2f>(i, j);
			cv::Point2f _newFlow;
			_newFlow.x = _flow.x + j;
			_newFlow.y = _flow.y + i;
			updateflow.at<cv::Point2f>(i, j) = _newFlow;
		}
	}
	cv::remap(asapPtr_thirditer->getFlowfield(), refineflow, updateflow, cv::Mat(), cv::INTER_LINEAR);
#ifdef _DEBUG_TINY_STITCHER
	cv::Mat visual;
	cv::remap(localImg, warpImg_fourthiter, refineflow, cv::Mat(), cv::INTER_LINEAR);
#endif
	// resample final feature points
	int meshrows = 16;
	int meshcols = 16;
	refPts_fourthiter = refPts_thirditer;
	localPts_fourthiter = localPts_thirditer;
	resampleFeaturePoints(refineflow, asapPtr_thirditer->getFlowfield(),
		refPts_fourthiter, localPts_fourthiter, meshrows, meshcols);
#ifdef _DEBUG_TINY_STITCHER
	visual = TinyStitcher::visualMatchingPts(refBlk, localImg, refPts_fourthiter,
		localPts_fourthiter, 0);
#endif

	// apply asap to generate final warped image
	int width = refBlk.cols;
	int height = refBlk.rows;
	float quadWidth = static_cast<float>(width) / static_cast<float>(meshcols);
	float quadHeight = static_cast<float>(height) / static_cast<float>(meshrows);
	float smoothWeight = 0.2;
	asapPtr_fourthiter->setMesh(height, width, quadHeight, quadWidth, smoothWeight);
	asapPtr_fourthiter->setControlPoints(localPts_thirditer, refPts_thirditer);
	asapPtr_fourthiter->solve();
	asapPtr_fourthiter->calcFlowField();
	cv::Mat flow = asapPtr_fourthiter->getFlowfield();

	cv::remap(localImg, warpImg_fourthiter, flow, cv::Mat(), cv::INTER_LINEAR);
	return 0;
}

/**
@brief get final warped image
@return cv::Mat warpImg: finally warped image
*/
cv::Mat TinyStitcher::getFinalWarpImg() {
	return warpImg_fourthiter;
}

/**
@brief get reference block image
@return cv::Mat warpImg: reference block image
*/
cv::Mat TinyStitcher::getRefBlkImg() {
	return refBlk;
}

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
cv::Mat TinyStitcher::visualMatchingPts(cv::Mat img1, cv::Mat img2,
	std::vector<cv::Point2f> pt1, std::vector<cv::Point2f> pt2, int direction) {
	cv::Mat showImg;
	cv::Size showSize;
	if (direction == 0) {// horizontal
		showSize = cv::Size(img1.cols + img2.cols, img1.rows);
		showImg.create(showSize, CV_8UC3);
		cv::Rect rect(0, 0, img1.cols, img1.rows);
		img1.copyTo(showImg(rect));
		rect.x += img1.cols;
		img2.copyTo(showImg(rect));
	}
	else {// vertical
		showSize = cv::Size(img1.cols, img1.rows + img2.rows);
		showImg.create(showSize, CV_8UC3);
		cv::Rect rect(0, 0, img1.cols, img1.rows);
		img1.copyTo(showImg(rect));
		rect.y += img1.rows;
		img2.copyTo(showImg(rect));
	}
	cv::RNG rng(12345);
	int r = 14;
	for (int i = 0; i < pt1.size(); i++) {
		cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		cv::Point2f p1 = pt1[i];
		cv::Point2f p2 = pt2[i];
		if (p1.x < 0 || p1.x >= img1.cols || p1.y < 0 || p1.y >= img1.rows)
			continue;
		if (p2.x < 0 || p2.x >= img2.cols || p2.y < 0 || p2.y >= img2.rows)
			continue;
		if (direction == 0)
			p2.x += img1.cols;
		else p2.y += img1.rows;
		circle(showImg, p1, r, color, -1, 8, 0);
		circle(showImg, p2, r, color, -1, 8, 0);
	}
	return showImg;
}

int TinyStitcher::find_position(cv::Mat refImg, cv::Mat localImg, cv::Point &out_point)
{
	cv::Mat refEdge = this->SEDDetector(refImg, 0.5);
	cv::Mat localEdge = this->SEDDetector(localImg, 0.5);
	// resize localview image
	cv::Mat templ, templEdge;
	sizeBlock = cv::Size(localImg.cols * scale, localImg.rows * scale);
	cv::resize(localImg, templ, sizeBlock);
	cv::resize(localEdge, templEdge, sizeBlock);

	//colorCorrectRGB(templ, refImg);    ///////////////////////
	cv::Mat result, resultEdge;
	cv::matchTemplate(refImg, templ, result, cv::TM_CCOEFF_NORMED);
	cv::matchTemplate(refEdge, templEdge, resultEdge, cv::TM_CCOEFF_NORMED);
	result = result.mul(resultEdge);

	cv::Point maxLoc;
	cv::minMaxLoc(result, NULL, NULL, NULL, &maxLoc);
	out_point = maxLoc;

	return 0;
}

int TinyStitcher::gimble_find_position(cv::Mat refImg, cv::Mat localImg, cv::Point ref_point, 
	                               float region_mul,cv::Point &out_point)     //region_mul must bigger than 1,reginal:243*182
{
	int region_width, region_height;
	cv::Rect imgrect(0, 0, refImg.cols, refImg.rows);
	cv::Size origin_size(refImg.cols*scale, refImg.rows*scale);
	cv::Size new_size(refImg.cols*scale*region_mul, refImg.rows*scale*region_mul);
	cv::Point2f new_point = cv::Point2f(ref_point.x - (new_size.width - origin_size.width) / 2,
		ref_point.y - (new_size.height - origin_size.height) / 2);
	cv::Rect new_region;
	cv::Mat new_ref;

	if (isInside(new_point, imgrect) && isInside(cv::Point2f(new_point.x+new_size.width, new_point.y + new_size.height), imgrect))
	{
		new_region=cv::Rect(new_point, new_size);
	}
	else
	{
		out_point = ref_point;
		return -1;
	}
	refImg(new_region).copyTo(new_ref);

	
	cv::Mat refEdge = this->SEDDetector(new_ref, 0.5);
	cv::Mat localEdge = this->SEDDetector(localImg, 0.5);
	// resize localview image
	cv::Mat templ, templEdge;
	sizeBlock = cv::Size(localImg.cols * scale, localImg.rows * scale);
	cv::resize(localImg, templ, sizeBlock);
	cv::resize(localEdge, templEdge, sizeBlock);
	//colorCorrectRGB(templ, new_ref);    ///////////////////////

	cv::Mat result, resultEdge;
	cv::matchTemplate(new_ref, templ, result, cv::TM_CCOEFF_NORMED);
	cv::matchTemplate(refEdge, templEdge, resultEdge, cv::TM_CCOEFF_NORMED);
	result = result.mul(resultEdge);

	cv::Point maxLoc;
	cv::minMaxLoc(result, NULL, NULL, NULL, &maxLoc);
	out_point.x = maxLoc.x + new_point.x;
	out_point.y = maxLoc.y + new_point.y;

	std::cout << "This current x is:" << out_point.x << "    y is:" << out_point.y << std::endl;

	return 0;
}

int TinyStitcher::init_stitcher()
{
	scale = 0.118;
	ptr = cv::ximgproc::createStructuredEdgeDetection("E:/code/project/XIMEA_giga_gimble/model/model.yml");
	return 0;
}


cv::Mat TinyStitcher::merge_copy(cv::Mat src)
{
	std::vector<cv::Mat> temp(3);
	cv::Mat dst;
	for (int i = 0; i < 3; i++)
	{
		src.copyTo(temp[i]);
	}
	merge(temp, dst);
	return dst;
}

int TinyStitcher::calibration_record(float start_position, float end_position)
{
	// init buffer
	CSerial serial;
	TinyStitcher stitcher;

	cv::Mat watching1, watching2, watcging3;
	std::vector<cam::Imagedata> imgdatas(2);
	std::vector<cv::Mat> imgs(2);
	// init camera
	std::vector<cam::GenCamInfo> camInfos;
	std::shared_ptr<cam::GenCamera> cameraPtr
		= cam::createCamera(cam::CameraModel::XIMEA_xiC);
	cameraPtr->init();
	// set camera setting
	cameraPtr->startCapture();
	cameraPtr->setFPS(-1, 20);
	cameraPtr->setAutoExposure(-1, cam::Status::on);
	cameraPtr->setAutoExposureLevel(-1, 40);
	cameraPtr->setAutoWhiteBalance(-1);
	cameraPtr->makeSetEffective();
	// set capturing setting
	cameraPtr->setCamBufferType(cam::GenCamBufferType::Raw);
	cameraPtr->setCaptureMode(cam::GenCamCaptureMode::Continous, 40);
	cameraPtr->setCapturePurpose(cam::GenCamCapturePurpose::Streaming);
	//cameraPtr->setVerbose(true);
	// get camera info
	cameraPtr->getCamInfos(camInfos);
	cameraPtr->startCaptureThreads();
	cameraPtr->makeSetEffective();
	//storage
	char name[200], temp[50];
	cv::Point start_point, end_point;
	std::ofstream myfile("E:/code/project/camdriver/test_new/data.txt", std::ios::out | std::ios::app);
	// capture frames
	for (int k = 0; k<5; k++)
	{
		serial.Serial_Send_Yaw(start_position);
		Sleep(1000);
		cameraPtr->captureFrame(imgdatas);
		imgs[0] = cv::Mat(camInfos[0].height, camInfos[0].width,
			CV_8U, reinterpret_cast<void*>(imgdatas[0].data));
		imgs[1] = cv::Mat(camInfos[1].height, camInfos[1].width,
			CV_8U, reinterpret_cast<void*>(imgdatas[1].data));

		stitcher.find_position(merge_copy(imgs[1]), merge_copy(imgs[0]), start_point);

		//watching2 = imgs[1].clone();
		strcpy(name, "E:/code/project/camdriver/test_new/local_");
		_itoa(start_position, temp, 10);
		strcat(name, temp);
		strcat(name, "_");
		_itoa(k, temp, 10);
		strcat(name, temp);
		strcat(name, ".png");
		cv::imwrite(name, imgs[0]);
		Sleep(1000);

		serial.Serial_Send_Yaw(end_position);
		Sleep(1000);
		cameraPtr->captureFrame(imgdatas);
		imgs[0] = cv::Mat(camInfos[0].height, camInfos[0].width,
			CV_8U, reinterpret_cast<void*>(imgdatas[0].data));
		imgs[1] = cv::Mat(camInfos[1].height, camInfos[1].width,
			CV_8U, reinterpret_cast<void*>(imgdatas[1].data));

		stitcher.find_position(merge_copy(imgs[1]), merge_copy(imgs[0]), end_point);
		strcpy(name, "E:/code/project/camdriver/test_new/local_");
		_itoa(end_position, temp, 10);
		strcat(name, temp);
		strcat(name, "_");
		_itoa(k, temp, 10);
		strcat(name, temp);
		strcat(name, ".png");
		cv::imwrite(name, imgs[0]);

		/*strcpy(name, "D:/code/project/camdriver/test/global150_");
		_itoa(k, temp, 10);
		strcat(name, temp);
		strcat(name, ".png");
		cv::imwrite(name, imgs[1]);*/

		myfile << "local_" << end_position << " - local_" << start_position << "    x:" << end_point.x - start_point.x
			<< "    y:" << end_point.y - start_point.y << std::endl;
		Sleep(1000);
		printf("%d", k);
	}
	myfile.close();
	cameraPtr->stopCaptureThreads();
	cameraPtr->release();
	return 0;
}

int TinyStitcher::test_100(cv::Point2f src,CSerial serial)
{
	TinyStitcher stitcher;
	cv::Mat watching1, watching2, watcging3;
	cv::Point start_point, end_point;
	cv::Point2f dst_pulse;
	std::vector<cam::Imagedata> imgdatas(2);
	std::vector<cv::Mat> imgs(2);
	// init camera
	std::vector<cam::GenCamInfo> camInfos;
	std::shared_ptr<cam::GenCamera> cameraPtr
		= cam::createCamera(cam::CameraModel::XIMEA_xiC);
	cameraPtr->init();
	// set camera setting
	cameraPtr->startCapture();
	cameraPtr->setFPS(-1, 20);
	cameraPtr->setAutoExposure(-1, cam::Status::on);
	cameraPtr->setAutoExposureLevel(-1, 40);
	cameraPtr->setAutoWhiteBalance(-1);
	cameraPtr->makeSetEffective();
	// set capturing setting
	cameraPtr->setCamBufferType(cam::GenCamBufferType::Raw);
	cameraPtr->setCaptureMode(cam::GenCamCaptureMode::Continous, 40);
	cameraPtr->setCapturePurpose(cam::GenCamCapturePurpose::Streaming);
	//cameraPtr->setVerbose(true);
	// get camera info
	cameraPtr->getCamInfos(camInfos);
	cameraPtr->startCaptureThreads();
	cameraPtr->makeSetEffective();

	cameraPtr->captureFrame(imgdatas);
	cv::Mat(camInfos[0].height, camInfos[0].width,
		CV_8U, reinterpret_cast<void*>(imgdatas[0].data)).copyTo(imgs[0]);
	cv::Mat(camInfos[1].height, camInfos[1].width,
		CV_8U, reinterpret_cast<void*>(imgdatas[1].data)).copyTo(imgs[1]);

	stitcher.find_position(merge_copy(imgs[1]), merge_copy(imgs[0]), start_point);
	std::cout << "start_x:" << start_point.x << "    start_y:" << start_point.y << std::endl;
	//move(dst_pulse);
	std::cout << "pulse:" << dst_pulse.x << dst_pulse.y << std::endl;
	serial.Serial_Send_Yaw(dst_pulse.x);
	serial.Serial_Send_Pitch(dst_pulse.y);

	Sleep(1000);

	cameraPtr->captureFrame(imgdatas);
	cv::Mat(camInfos[0].height, camInfos[0].width,
		CV_8U, reinterpret_cast<void*>(imgdatas[0].data)).copyTo(imgs[0]);
	cv::Mat(camInfos[1].height, camInfos[1].width,
		CV_8U, reinterpret_cast<void*>(imgdatas[1].data)).copyTo(imgs[1]);
	stitcher.find_position(merge_copy(imgs[1]), merge_copy(imgs[0]), end_point);
	std::cout << "end_x:" << end_point.x << "    end_y:" << end_point.y << std::endl;
	system("pause");
	return 0;
}

//int TinyStitcher::move(cv::Point& dst)       //根据类中的当前点和当前脉冲使云台移动到目标像素位置
//{
//	float pulse_delta;
//	cv::Point2f dst_pulse;
//	pulse_delta = (dst.y - current_point.y) / 3.0;
//	dst_pulse.y = current_pulse.y - pulse_delta;
//	if (dst_pulse.y >= PM + 120 - dY)    //云台转动超出全景图范围
//	{
//		dst_pulse.y = PM + 120 - 1.1 * dY;
//		pulse_delta = current_pulse.y - dst_pulse.y;
//		dst.y = pulse_delta * 3 + current_point.y;
//		std::cout << "dst_pulse.y >= PM + 120" << std::endl;
//	}
//	else if (dst_pulse.y <= PM + 120 - dY*(Row-1))
//	{
//		dst_pulse.y = PM + 120 - dY*(Row - 1.1);
//		pulse_delta = dst_pulse.y - current_pulse.y;
//		dst.y = pulse_delta * 3 + current_point.y;
//		std::cout << "dst_pulse.y <= PM + 120 - dY*Row" << std::endl;
//	}
//
//	pulse_delta = (dst.x - current_point.x) / 4.0;  //3.0
//	dst_pulse.x = current_pulse.x - pulse_delta;
//	if (dst_pulse.x >= YM + 180 - dX)
//	{
//		dst_pulse.x = YM + 180-1.1*dX;
//		pulse_delta = current_pulse.x - dst_pulse.x;
//		dst.x = pulse_delta * 4 + current_point.x;
//		std::cout << "dst_pulse.x >= YM + 180" << std::endl;
//	}
//	else if(dst_pulse.x <= YM + 180 - dX*(Col-1))
//	{
//		dst_pulse.x = YM + 180 - dX*(Col - 1.1);
//		pulse_delta = current_pulse.x - dst_pulse.x;
//		dst.x = pulse_delta * 4 + current_point.x;
//		std::cout << "dst_pulse.x <= YM + 180 - dX*Col" << std::endl;
//	}
//	//if (dst_pulse.y >= PM + 120)    //云台转动超出全景图范围
//	//{
//	//	dst_pulse.y = PM + 120 - dY;
//	//	pulse_delta = current_pulse.y - dst_pulse.y;
//	//	dst.y = pulse_delta * 3 + current_point.y;
//	//	std::cout << "dst_pulse.y >= PM + 120" << std::endl;
//	//}
//	//else if (dst_pulse.y <= PM + 120 - dY*Row)
//	//{
//	//	dst_pulse.y = PM + 120 - dY*(Row - 1);
//	//	pulse_delta = dst_pulse.y - current_pulse.y;
//	//	dst.y = pulse_delta * 3 + current_point.y;
//	//	std::cout << "dst_pulse.y <= PM + 120 - dY*Row" << std::endl;
//	//}
//
//	//pulse_delta = (dst.x - current_point.x) / 4.0;  //3.0
//	//dst_pulse.x = current_pulse.x - pulse_delta;
//	//if (dst_pulse.x >= YM + 180 )
//	//{
//	//	dst_pulse.x = YM + 180-dX;
//	//	pulse_delta = current_pulse.x - dst_pulse.x;
//	//	dst.x = pulse_delta * 4 + current_point.x;
//	//	std::cout << "dst_pulse.x >= YM + 180" << std::endl;
//	//}
//	//else if(dst_pulse.x <= YM + 180 - dX*Col)
//	//{
//	//	dst_pulse.x = YM + 180 - dX*(Col - 1);
//	//	pulse_delta = current_pulse.x - dst_pulse.x;
//	//	dst.x = pulse_delta * 4 + current_point.x;
//	//	std::cout << "dst_pulse.x <= YM + 180 - dX*Col" << std::endl;
//	//}
//	current_pulse.x = dst_pulse.x;
//	current_pulse.y = dst_pulse.y;
//
//	std::cout << current_pulse << std::endl;
//	current_point.x = dst.x;
//	current_point.y = dst.y;
//	return 0;
//}

//void TinyStitcher::global2pano(cv::Mat global, cv::Mat pano)
//{
//	cv::Rect roi(1000, 500, pano.cols - 1500, pano.rows - 1000);
//	cv::Mat pano_cut;
//	pano(roi).copyTo(pano_cut);
//
//
//}

int TinyStitcher::move(cv::Point& dst)       //根据类中的当前点和当前脉冲使云台移动到目标像素位置
{
	float pulse_delta;
	cv::Point2f dst_pulse;
	pulse_delta = (dst.y - current_point.y) / 3.0;
	dst_pulse.y = current_pulse.y - pulse_delta;
	current_pulse.y = dst_pulse.y;

	pulse_delta = (dst.x - current_point.x) / 4.0;  //3.0
	dst_pulse.x = current_pulse.x - pulse_delta;
	current_pulse.x = dst_pulse.x;

	std::cout << current_pulse << std::endl;
	current_point.x = dst.x;
	current_point.y = dst.y;
	return 0;
}

int TinyStitcher::detect_move(cv::Point dst)       //根据类中的当前点和当前脉冲使云台移动到目标像素位置
{
	float pulse_delta;
	cv::Point2f dst_pulse;
	pulse_delta = (dst.y - current_point.y) / 3.0;
	dst_pulse.y = current_pulse.y - pulse_delta;
	if (dst_pulse.y >= PM + 120 - dY)    //云台转动超出全景图范围
	{
		return -1;
		std::cout << "dst_pulse.y >= PM + 120" << std::endl;
	}
	else if (dst_pulse.y <= PM + 120 - dY*(Row - 1))
	{
		return -1;
		std::cout << "dst_pulse.y <= PM + 120 - dY*Row" << std::endl;
	}

	pulse_delta = (dst.x - current_point.x) / 4.0;  //3.0
	dst_pulse.x = current_pulse.x - pulse_delta;
	if (dst_pulse.x >= YM + 180 - dX)
	{
		return -1;
		std::cout << "dst_pulse.x >= YM + 180" << std::endl;
	}
	else if (dst_pulse.x <= YM + 180 - dX*(Col - 1))
	{
		return -1;
		std::cout << "dst_pulse.x <= YM + 180 - dX*Col" << std::endl;
	}
	return 0;
}
