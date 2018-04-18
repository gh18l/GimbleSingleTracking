/**
@brief feature matching
@author: Shane Yuan
@data: Dec 11, 2017
*/

#include "FeatureMatch.h"

FeatureMatch::FeatureMatch(): patchSize(cv::Size(256, 256)), 
	searchSize(cv::Size(512, 512)) {}
FeatureMatch::~FeatureMatch() {}

/**
@brief set size
@param cv::Size patchSize: size of patch (template)
@param cv::Size searchSize: size of searching range
@return int
*/
int FeatureMatch::setSize(cv::Size patchSize, cv::Size searchSize) {
	this->patchSize = patchSize;
	this->searchSize = searchSize;
	return 0;
}

/**
@brief detect ket points (using goodfeature2track algorithm)
@param cv::Mat img: input image
@param std::vector<cv::Point2f> & keypts: output key points
@return int
*/
int FeatureMatch::detectKeyPts(cv::Mat img, std::vector<cv::Point2f> & keypts, float scale) {
	cv::Mat imgGray;
	cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);
	// resize image
	cv::Size smallsize(imgGray.cols * scale, imgGray.rows * scale);
	cv::resize(imgGray, imgGray, smallsize);
	// apply key points detector
	int maxPts = 2000;
	int mindist = 50.0f * scale;
	cv::goodFeaturesToTrack(imgGray, keypts, maxPts, 0.01, mindist, cv::Mat(), 5, false, 0.01);
	for (size_t i = 0; i < keypts.size(); i++) {
		keypts[i] = keypts[i] / scale;
	}
	return 0;
}

/**
@brief check if a point is in a rect
@param cv::Point2f pt: input point
@param cv::Rect rect: input rect
@return bool: if inside rectagnle
*/
bool FeatureMatch::isInside(cv::Point2f pt, cv::Rect rect) {
	if (rect.contains(pt))
		return true;
	else return false;
}

/**
@brief calculate feature points between images
@param cv::Mat refBlk: reference block
@param cv::Mat localImg: local image
@param std::vector<cv::Point2f> refPts: reference points
@param std::vector<cv::Point2f> localPts: localview points
@return int
*/
int FeatureMatch::match(cv::Mat refBlk, cv::Mat localImg, std::vector<cv::Point2f> refPts,
	std::vector<cv::Point2f> localPts, std::vector<cv::Point2f> & outRefPts,
	std::vector<cv::Point2f> & outLocalPts) {
	outRefPts.clear();
	outLocalPts.clear();
	size_t len = refPts.size();
	cv::Rect imgrect(0, 0, refBlk.cols, refBlk.rows);
	for (size_t i = 0; i < len; i++) {
		cv::Point2f patchpt = localPts[i];
		cv::Point2f searchpt = refPts[i];
		cv::Point2f patch_tl = cv::Point2f(patchpt.x - patchSize.width / 2,
			patchpt.y - patchSize.height / 2);
		cv::Point2f patch_br = cv::Point2f(patchpt.x + patchSize.width - 1,
			patchpt.y + patchSize.height - 1);
		cv::Point2f search_tl = cv::Point2f(searchpt.x - searchSize.width / 2,
			searchpt.y - searchSize.height / 2);
		cv::Point2f search_br = cv::Point2f(search_tl.x + searchSize.width - 1,
			search_tl.y + searchSize.height - 1);
		if (isInside(search_tl, imgrect) && isInside(search_br, imgrect)
			&& isInside(patch_tl, imgrect) && isInside(patch_br, imgrect)) {
			// apply matching template
			cv::Rect patch_rect = cv::Rect(patch_tl, patch_br);
			cv::Rect search_rect = cv::Rect(search_tl, search_br);
			cv::Mat patchImg = localImg(patch_rect);
			cv::Mat searchImg = refBlk(search_rect);
			cv::Mat result;
			cv::matchTemplate(searchImg, patchImg, result, cv::TM_CCOEFF_NORMED);
			cv::Point maxLoc;
			cv::minMaxLoc(result, NULL, NULL, NULL, &maxLoc);
			cv::Point2f newpt = cv::Point2f(maxLoc.x + search_rect.x + patchSize.width / 2,
				maxLoc.y + search_rect.y + patchSize.height / 2);
			outRefPts.push_back(newpt);
			outLocalPts.push_back(patchpt);
		}
	}
	return 0;
}

/**
@brief global ransac to discard outliers
@param std::vector<cv::Point2f> & refPts: input/output reference points
@param std::vector<cv::Point2f> & localPts: input/output localview points
@param float threshold: threshold used in ransac
@return int
*/
int FeatureMatch::gloablRansac(std::vector<cv::Point2f> & refPts,
	std::vector<cv::Point2f> & localPts, float threshold) {
	cv::Mat mask;
	cv::findHomography(localPts, refPts, CV_RANSAC, threshold, mask);
	std::vector<cv::Point2f> outRefPts;
	std::vector<cv::Point2f> outLocalPts;
	for (size_t i = 0; i < mask.rows; i++) {
		if (mask.at<uchar>(i, 0) != 0) {
			outRefPts.push_back(refPts[i]);
			outLocalPts.push_back(localPts[i]);
		}
	}
	return 0;
}

/**
@brief local ransac to discard outliers
@param std::vector<cv::Point2f> & refPts: input/output reference points
@param std::vector<cv::Point2f> & localPts: input/output localview points
@param int width: width of image
@param int height: height of image
@param
@param float threshold: threshold used in ransac
@return int
*/
int FeatureMatch::localRansac(std::vector<cv::Point2f> & refPts, std::vector<cv::Point2f> & localPts,
	int width, int height, int rows, int cols, float threshold) {
	std::vector<std::vector<cv::Point2f>> localKp1;
	std::vector<std::vector<cv::Point2f>> localKp2;
	localKp1.resize(rows * cols);
	localKp2.resize(rows * cols);
	// divide
	int lens = refPts.size();
	float quadWidth = width / cols;
	float quadHeight = height / rows;
	for (int ind = 0; ind < lens; ind++) {
		int i = static_cast<int>(refPts[ind].y / quadHeight);
		int j = static_cast<int>(refPts[ind].x / quadWidth);
		int dist = std::abs(refPts[ind].y - localPts[ind].y) + std::abs(refPts[ind].x - localPts[ind].x);
		if (dist < 1000) {
			int kpInd = i * cols + j;
			localKp1[kpInd].push_back(refPts[ind]);
			localKp2[kpInd].push_back(localPts[ind]);
		}
	}
	// local ransac
	refPts.clear();
	localPts.clear();
	for (int i = 0; i < rows * cols; i++) {
		cv::Mat mask;
		if (localKp1[i].size() >= 4) {
			cv::Mat H = findHomography(localKp1[i], localKp2[i], cv::RANSAC, threshold, mask);
			for (int j = 0; j < mask.rows; j++) {
				if (mask.at<uchar>(j, 0) == 1) {
					refPts.push_back(localKp1[i][j]);
					localPts.push_back(localKp2[i][j]);
				}
			}
		}
	}
	return 0;
}

/**
@brief build correspondecne for two input images
@param cv::Mat refBlk: input low quality reference image
@param cv::Mat localImg: input high quality localview image
@param float scale: scale used for feature matching (directly operating on
original scale is slow and time consuming)
@return int
*/
int FeatureMatch::buildCorrespondence(cv::Mat refBlk, cv::Mat localImg, float scale,
	std::vector<cv::Point2f> & outRefPts, std::vector<cv::Point2f> & outLocalPts) {
	// resize image first
	cv::Mat refBlk_small;
	cv::Mat localImg_small;
	cv::Size size_large = refBlk.size();
	cv::Size size_small = cv::Size(size_large.width * scale, size_large.height * scale);
	cv::resize(refBlk, refBlk_small, size_small);
	cv::resize(localImg, localImg_small, size_small);
	// detect key points
	std::vector<cv::Point2f> localPts;
	std::vector<cv::Point2f> refPts;
	detectKeyPts(localImg_small, localPts, 1.0f);   //0.25f
	refPts = localPts;
	// build correspondence
	match(refBlk_small, localImg_small, refPts, localPts, outRefPts, outLocalPts);
	// global ransac
	gloablRansac(outRefPts, outLocalPts, 3.0f);
	// apply scale
	for (size_t i = 0; i < outRefPts.size(); i++) {
		outRefPts[i] = outRefPts[i] / scale;
		outLocalPts[i] = outLocalPts[i] / scale;
	}
	return 0;
}

/**
@brief build correspondecne for two input images (roughly matched key points provided)
@param cv::Mat refBlk: input low quality reference image
@param cv::Mat localImg: input high quality localview image
@param float scale: scale used for feature matching (directly operating on
original scale is slow and time consuming)
@param cv::Mat globalH: gloabl homography matrix
@param std::vector<cv::Point2f> & outRefPts: output reference feature points
@param std::vector<cv::Point2f> & outLocalPts: output local feature points
@return int
*/
int FeatureMatch::buildCorrespondence(cv::Mat refBlk, cv::Mat localImg, float scale, cv::Mat globalH,
	std::vector<cv::Point2f> & outRefPts, std::vector<cv::Point2f> & outLocalPts) {
	// resize image first
	cv::Mat refBlk_small;
	cv::Mat localImg_small;
	cv::Size size_large = refBlk.size();
	cv::Size size_small = cv::Size(size_large.width * scale, size_large.height * scale);
	cv::resize(refBlk, refBlk_small, size_small);
	cv::resize(localImg, localImg_small, size_small);
	// detect key points
	std::vector<cv::Point2f> localPts;
	std::vector<cv::Point2f> refPts;
	detectKeyPts(localImg_small, localPts, 0.25f);
	// prewarp 
	globalH.at<float>(0, 2) = globalH.at<float>(0, 2) * scale;
	globalH.at<float>(1, 2) = globalH.at<float>(1, 2) * scale;
	for (size_t i = 0; i < localPts.size(); i++) {
		cv::Point2f pt = localPts[i];
		// calculate new position
		float new_x = globalH.at<float>(0, 0) * pt.x + globalH.at<float>(0, 1) * pt.y
			+ globalH.at<float>(0, 2);
		float new_y = globalH.at<float>(1, 0) * pt.x + globalH.at<float>(1, 1) * pt.y
			+ globalH.at<float>(1, 2);
		float new_z = globalH.at<float>(2, 0) * pt.x + globalH.at<float>(2, 1) * pt.y
			+ globalH.at<float>(2, 2);
		refPts.push_back(cv::Point2f(new_x / new_z, new_y / new_z));
	}
	// build correspondence
	match(refBlk_small, localImg_small, refPts, localPts, outRefPts, outLocalPts);
	// apply scale
	for (size_t i = 0; i < outRefPts.size(); i++) {
		outRefPts[i] = outRefPts[i] / scale;
		outLocalPts[i] = outLocalPts[i] / scale;
	}
	// local ransac
	localRansac(outRefPts, outLocalPts, refBlk.cols, refBlk.rows, 4, 4, 3.0f);
	return 0;
}