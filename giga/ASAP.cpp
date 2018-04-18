/**
 * C++ source file of as similar as possible warping class
 *
 * Author: Shane Yuan
 * Date: Jan 13, 2016
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <time.h>

// opencv
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

#include "ASAP.h"

ASAP::ASAP() {
}

ASAP::ASAP(int height, int width, float quadHeight, float quadWidth, float weight) {
	this->setMesh(height, width, quadHeight, quadWidth, weight);
}

int ASAP::setMesh(int height, int width, float quadHeight, float quadWidth, float weight) {
	this->source = Mesh(height, width, quadWidth, quadHeight);
	this->target = Mesh(height, width, quadWidth, quadHeight);

	this->imgHeight = height;
	this->imgWidth = width;
	this->quadHeight = quadHeight;
	this->quadWidth = quadWidth;
	this->height = source.getMeshHeight();
	this->width = source.getMeshWidth();

	this->numSmoothCons = ((this->height - 2) * (this->width - 2) * 16 + (2 * (this->width + this->height) - 8) * 8 + 4 * 4);
	this->columns = this->width * this->height * 2;

	this->smoothConstraints = Eigen::MatrixXf::Zero(this->numSmoothCons * 5, 3);
	this->SCcount = 0;

	this->createSmoothCons(weight);
#ifdef MY_DEBUG
	MyUtility::dumpEigenMatrix(this->smoothConstraints, "smoothconstraints.txt");
#endif
	return 0;
}

ASAP::~ASAP() {
	
};


/** set control points
 * Input:
 * vector<Point2f> inputsPts : original SIFT (or other feature points) points coordinates
 * vector<Point2f> outputsPts : target SIFT (or other feature points) points coordinates
 * vector<float> ptWeights: weights of input points
 * Return
 * int : ErrorCode
 */
int ASAP::setControlPoints(vector<Point2f> inputsPts, vector<Point2f> outputsPts, std::vector<float> ptWeight /* = std::vector<float>()*/) {
	int len = outputsPts.size();

	this->orgPt.clear();
	this->desPt.clear();
	this->dataI.clear();
	this->dataJ.clear();
	this->dataV00.clear();
	this->dataV01.clear();
	this->dataV10.clear();
	this->dataV11.clear();

	for (int i = 0; i < len; i ++) {
		Point2f pt = outputsPts[i];
		if (std::min(std::min(pt.x, pt.y), std::min(imgHeight - pt.y, imgWidth - pt.x)) < 20) {
			continue;
		}
		this->orgPt.push_back(pt);
		this->desPt.push_back(inputsPts[i]);
		if (ptWeight.size() > 0)
			this->ptWeight.push_back(ptWeight[i]);
		else this->ptWeight.push_back(1.0f);

		dataI.push_back((int)(pt.y / this->quadHeight) + 1);    //表示在第几个纵向格子内
		dataJ.push_back((int)(pt.x / this->quadWidth) + 1);    //表示在第几个横向格子内
//		cout << i << "\t" << pt << endl;

		Quad qd = this->source.getQuad(dataI[dataI.size() - 1], dataJ[dataJ.size() - 1]);  //表示四个顶点的图像坐标
		BilinearCoeff coef = qd.getBilinearCoeff(pt);
		dataV00.push_back(coef.coeff(0, 0));
		dataV01.push_back(coef.coeff(1, 0));
		dataV10.push_back(coef.coeff(2, 0));
		dataV11.push_back(coef.coeff(3, 0));
	}

	return 0;
}

/**
* @brief fusionCorrespondence using feature points on detail image, feature point flow and pixel based flow
* @param std::vector<cv::Point2f> detailPts: input feature points on detail images
* @param std::vector<cv::Point2f> refPts: input feature points on detail images
* @param cv::Mat featurePointFlow: flowfield calculated from sparse feature points
* @param cv::Mat pixelValueFlow: flowfield calculated using variational refinement
* @param std::vector<cv::Point2f> & outDetailPts: output detail feature points
* @param std::vector<cv::Point2f> & outRefPts: output reference feature points
* @param std::vector<float> & outPtWeights: output point weights
* @return int: ErrorCode
*/
int ASAP::fusionCorrespondence(std::vector<cv::Point2f> detailPts, std::vector<cv::Point2f> refPts, cv::Mat featurePointFlow,
	cv::Mat pixelValueFlow, std::vector<cv::Point2f> & outDetailPts, std::vector<cv::Point2f> & outRefPts, std::vector<float> & outPtWeights) {
	// weight for different kinds of points
	float znccWeight = 5.0;
	float keypointWeight = 5.0;
	float centerWeight = 1.0;
	// calculate the distribution of detail feature points
	cv::Mat_<uint> distrubution = cv::Mat_<uint>::zeros(source.getMeshHeight() - 1, source.getMeshWidth() - 1);
	cv::Mat_<uint> distrubution2;
	for (int i = 0; i < detailPts.size(); i ++) {
		// get quad index
		int indI = static_cast<int>(detailPts[i].y / this->quadHeight);
		int indJ = static_cast<int>(detailPts[i].x / this->quadWidth);
		distrubution(indI, indJ) = distrubution(indI, indJ) + 1;
	}
	distrubution2 = distrubution.clone();
	// generate new feature points for remaining structure points
	std::vector<cv::Point2f> inputPts = outDetailPts;
	outDetailPts.clear();
	outRefPts.insert(outRefPts.end(), refPts.begin(), refPts.end());
	outDetailPts.insert(outDetailPts.end(), detailPts.begin(), detailPts.end());
	outPtWeights.resize(outDetailPts.size());
	for (int i = 0; i < outPtWeights.size(); i ++) {
		outPtWeights[i] = znccWeight;
	}
	int thresh = 3;
	for (int i = 0; i < inputPts.size(); i ++) {
		cv::Point2f pt1 = inputPts[i];
		int indI = static_cast<int>(pt1.y / this->quadHeight);
		int indJ = static_cast<int>(pt1.x / this->quadWidth);
		if (distrubution(indI, indJ) > thresh)
			continue;
		cv::Point2f pt2 = pixelValueFlow.at<Point2f>(pt1.y, pt1.x);
		if (abs(pt1.x - pt2.x) + abs(pt1.y - pt2.y) < 600) {
			float margin = 50;
			if (std::min(std::min(pt2.x, pt2.y), std::min(imgHeight - pt2.y, imgWidth - pt2.x)) > margin) {
				outRefPts.push_back(pt2);
				outDetailPts.push_back(pt1);
				outPtWeights.push_back(keypointWeight);
			}
		}
		distrubution2(indI, indJ) = distrubution2(indI, indJ) + 0.5;
	}
	// 
	for (int i = 0; i < distrubution.rows; i ++) {
		for (int j = 0; j < distrubution.cols; j ++) {
			// get for vertex coordinates
			cv::Point2f v00 = source.getVertex(i, j);
			cv::Point2f v01 = source.getVertex(i, j + 1);
			cv::Point2f v10 = source.getVertex(i + 1, j);
			cv::Point2f v11 = source.getVertex(i + 1, j + 1);
			cv::Mat flow;
			// generate feature points
			int num;
			if (!((distrubution2(i, j) > thresh + 1) || (distrubution(i, j) > thresh))) {
				if ((i == 0) || (j == 0) || (i == distrubution.rows - 1) || (j == distrubution.cols - 1))
					num = 2;
				else num = 1;
				for (float k = 0.5; k < num; k++)	{
					for (float l = 0.5; l < num; l++) {
						float x = static_cast<float>(k) / static_cast<float>(num)* v00.x
							+ (num - static_cast<float>(k)) / static_cast<float>(num)* v11.x;
						float y = static_cast<float>(l) / static_cast<float>(num)* v00.y
							+ (num - static_cast<float>(l)) / static_cast<float>(num)* v11.y;
						cv::Point2f pt1 = cv::Point2f(x, y);
						cv::Point2f pt2 = pixelValueFlow.at<Point2f>(pt1.y, pt1.x);
						if (abs(pt1.x - pt2.x) + abs(pt1.y - pt2.y) < 600) {
							float margin = 40;
							if (std::min(std::min(pt2.x, pt2.y), std::min(imgHeight - pt2.y, imgWidth - pt2.x)) > margin) {
								outRefPts.push_back(pt2);
								outDetailPts.push_back(pt1);
								outPtWeights.push_back(centerWeight);
							}
						}
					}
				}
			}			
		}
	}
	return 0;
}

/**
 * Solve as similar as possible warping
 */
int ASAP::solve() {
    // init b
	Eigen::MatrixXf b;
	createDataCons(b);
#ifdef MY_DEBUG
	MyUtility::dumpEigenMatrix(b, "b.txt");
	MyUtility::dumpEigenMatrix(this->dataConstraints, "dataConstraints.txt");
#endif
    // init A
	int n = smoothConstraints.rows() + dataConstraints.cols();
    Eigen::SparseMatrix<float> A(b.rows(), this->columns);
    typedef Eigen::Triplet<double> T;
    std::vector<T> tripletList;
    tripletList.reserve(n);
    for(int i = 0; i < smoothConstraints.rows(); i++) {
        tripletList.push_back(T(smoothConstraints(i, 0), smoothConstraints(i, 1), smoothConstraints(i, 2)));
    }
    for(int i = 0; i < dataConstraints.rows(); i++) {
        tripletList.push_back(T(dataConstraints(i, 0), dataConstraints(i, 1), dataConstraints(i, 2)));
    }
    A.setFromTriplets(tripletList.begin(), tripletList.end());
    // solve
    Eigen::SparseQR<Eigen::SparseMatrix<float>, Eigen::COLAMDOrdering<int> > solver;
    solver.compute(A);
    if(solver.info() != Eigen::Success) {
        // decomposition failed
        return -1;
    }
    Eigen::MatrixXf x = solver.solve(b);
    if(solver.info() != Eigen::Success) {
        // solving failed
        return -1;
    }
#ifdef MY_DEBUG
	MyUtility::dumpEigenMatrix(x, "x.txt");
#endif
    // remap x to point2f result
    int halfcolumns = this->columns / 2;
    for (int i = 0; i < this->height; i++) {
        for (int j = 0; j < this->width; j++) {
            Point2f  p = Point2f(x(i * width + j, 0), x(halfcolumns + i * width + j));
            target.setVertex(i, j, p);
        }
    }
	return 0;
}

/**
 * create smooth constrains
 * Input:
 * float weight: the weight of smooth term
 * Return:
 * int: ErrorCode
 */
int ASAP::createSmoothCons(float weight) {
	this->rowCount = 0;
	// compute coefficients on corner of the image
	int i = 0;
	int j = 0;
	this->addCoefficient(i, j, weight, 5);
	this->addCoefficient(i, j, weight, 6);
	i = 0; j = width - 1;
	this->addCoefficient(i, j, weight, 7);
	this->addCoefficient(i, j, weight, 8);
	i = height - 1; j = 0;
	this->addCoefficient(i, j, weight, 3);
	this->addCoefficient(i, j, weight, 4);
	i = height - 1; j = width - 1;
	this->addCoefficient(i, j, weight, 1);
	this->addCoefficient(i, j, weight, 2);
	// compute coefficients on border of the image
	i = 0;
	for (int j = 1; j <= width - 2; j++) {
		this->addCoefficient(i, j, weight, 5);
		this->addCoefficient(i, j, weight, 6);
		this->addCoefficient(i, j, weight, 7);
		this->addCoefficient(i, j, weight, 8);
	}
	i = height - 1;
	for (int j = 1; j <= width - 2; j++) {
		this->addCoefficient(i, j, weight, 1);
		this->addCoefficient(i, j, weight, 2);
		this->addCoefficient(i, j, weight, 3);
		this->addCoefficient(i, j, weight, 4);
	}
	j = 0;
	for (int i = 1; i <= height - 2; i++) {
		this->addCoefficient(i, j, weight, 3);
		this->addCoefficient(i, j, weight, 4);
		this->addCoefficient(i, j, weight, 5);
		this->addCoefficient(i, j, weight, 6);
	}
	j = width - 1;
	for (int i = 1; i <= height - 2; i++) {
		this->addCoefficient(i, j, weight, 1);
		this->addCoefficient(i, j, weight, 2);
		this->addCoefficient(i, j, weight, 7);
		this->addCoefficient(i, j, weight, 8);
	}

	for (int i = 1; i <= height - 2; i ++) {
		for (int j = 1; j <= width - 2; j ++) {
			this->addCoefficient(i, j, weight, 1);
			this->addCoefficient(i, j, weight, 2);
			this->addCoefficient(i, j, weight, 3);
			this->addCoefficient(i, j, weight, 4);
			this->addCoefficient(i, j, weight, 5);
			this->addCoefficient(i, j, weight, 6);
			this->addCoefficient(i, j, weight, 7);
			this->addCoefficient(i, j, weight, 8);
		}
	}

	return 0;
}

/**
 * create data constrains
 * Input:
 * Eigen::MatrixXf &: the result data constrian matrix
 * Return:
 * int: ErrorCode
 */
int ASAP::createDataCons(Eigen::MatrixXf & b) {
	int len = dataI.size();
	this->numDataCons = len * 2;
	b = Eigen::MatrixXf::Zero(this->numDataCons + this->numSmoothCons, 1);
	this->DCcount = 0;

	dataConstraints = Eigen::MatrixXf::Zero(len * 8, 3);

	for (int k = 0; k < len; k ++) {
		int i = dataI[k];
		int j = dataJ[k];
		float v00 = dataV00[k];
		float v01 = dataV01[k];
		float v10 = dataV10[k];
		float v11 = dataV11[k];

		dataConstraints(this->DCcount, 0) = this->rowCount;
		dataConstraints(this->DCcount, 1) = getXIndex((i - 1) * width + j - 1);
		dataConstraints(this->DCcount, 2) = v00;
		this->DCcount ++;
		dataConstraints(this->DCcount, 0) = this->rowCount;
		dataConstraints(this->DCcount, 1) = getXIndex((i - 1) * width + j);
		dataConstraints(this->DCcount, 2) = v01;
		this->DCcount++;
		dataConstraints(this->DCcount, 0) = this->rowCount;
		dataConstraints(this->DCcount, 1) = getXIndex(i * width + j - 1);
		dataConstraints(this->DCcount, 2) = v10;
		this->DCcount++;
		dataConstraints(this->DCcount, 0) = this->rowCount;
		dataConstraints(this->DCcount, 1) = getXIndex(i * width + j);
		dataConstraints(this->DCcount, 2) = v11;
		this->DCcount++;
		b(this->rowCount, 0) = desPt[k].x;
		this->rowCount++;

		dataConstraints(this->DCcount, 0) = this->rowCount;
		dataConstraints(this->DCcount, 1) = getYIndex((i - 1) * width + j - 1);
		dataConstraints(this->DCcount, 2) = v00;
		this->DCcount++;
		dataConstraints(this->DCcount, 0) = this->rowCount;
		dataConstraints(this->DCcount, 1) = getYIndex((i - 1) * width + j);
		dataConstraints(this->DCcount, 2) = v01;
		this->DCcount++;
		dataConstraints(this->DCcount, 0) = this->rowCount;
		dataConstraints(this->DCcount, 1) = getYIndex(i * width + j - 1);
		dataConstraints(this->DCcount, 2) = v10;
		this->DCcount++;
		dataConstraints(this->DCcount, 0) = this->rowCount;
		dataConstraints(this->DCcount, 1) = getYIndex(i * width + j);
		dataConstraints(this->DCcount, 2) = v11;
		this->DCcount++;
		b(this->rowCount, 0) = desPt[k].y;
		this->rowCount++;
#ifdef MY_DEBUG
		cout << "DCcount = " << DCcount << ", rowCount = " << rowCount << endl;
#endif
	}

	return 0;
}

/**
* @brief get output flowfield
* @return cv::Mat: output flowfield
*/
cv::Mat ASAP::getFlowfield() {
	return flowfield;
}

/**
* @brief get the deformation score of the mesh
* @return cv::Mat: output deformation score
*/
cv::Mat ASAP::getDeformationScore() {
	target.calcDeformationScore();
	cv::Mat shape = target.getDeformationScore();
	cv::Mat score = cv::Mat::zeros(shape.rows, shape.cols, CV_32F);
	for (int i = 0; i < shape.rows; i++) {
		for (int j = 0; j < shape.cols; j++) {
			cv::Vec4f center = shape.at<cv::Vec4f>(i, j);
			float up, down, left, right;
			if (i > 0)
				up = cv::norm(shape.at<cv::Vec4f>(i - 1, j), center);
			else up = 0;
			if (i < shape.rows - 1)
				down = cv::norm(shape.at<cv::Vec4f>(i + 1, j), center);
			else down = 0;
			if (j > 0)
				left = cv::norm(shape.at<cv::Vec4f>(i, j - 1), center);
			else left = 0;
			if (j < shape.cols - 1)
				right = cv::norm(shape.at<cv::Vec4f>(i, j + 1), center);
			else right = 0;
			score.at<float>(i, j) = std::max(std::max(up, down), std::max(left, right));
		}
	}
	this->score = score;
	this->shapeMat = shape;
	return score;
}

/**
* @brief calculate the deformation score and draw mask
* @param cv::Mat & mask: drawn mask 
* @return cv::Mat: output deformation score
*/
int ASAP::getDeformationMask(cv::Mat & mask, cv::Mat & score, float th /* = 0.25 */) {
	getDeformationScore();
	mask = cv::Mat::zeros(imgHeight, imgWidth, CV_32F);
	score = this->score;

	for (int i = 0; i < score.rows; i++) {
		for (int j = 0; j < score.cols; j ++) {
			if (score.at<float>(i, j) < th)
				continue;
			Quad q = source.getQuad(i + 1, j + 1);
			for (int ii = q.getVertex(0, 0).y; ii <= q.getVertex(1, 1).y; ii ++) {
				for (int jj = q.getVertex(0, 0).x; jj <= q.getVertex(1, 1).x; jj++) {
					mask.at<float>(ii, jj) = 1;
				}
			}
		}
	}

	// smooth
	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(30, 30));
	cv::dilate(mask, mask, kernel);
	cv::Mat smoothK = cv::getGaussianKernel(30, 20, CV_32F);
	cv::Mat smoothK2D = smoothK * smoothK.t();
	cv::filter2D(mask, mask, -1, smoothK2D, cv::Point(-1, -1), 0.0, cv::BORDER_REPLICATE);
	return 0;
}

/**
* @brief write ASAP into file (only mesh is written into file in this function now)
* @param std::string filename
* @return int: ErrorCode 
*/
int ASAP::writeASAPMesh(std::string filename) {
	cv::FileStorage fs(filename, cv::FileStorage::WRITE);
	fs << "height" << height;
	fs << "width" << width;
	fs << "quadHeight" << quadHeight;
	fs << "quadWidth" << quadWidth;
	fs << "imgHeight" << imgHeight;
	fs << "imgWidth" << imgWidth;
	std::string sourcefilename = cv::format("%s.sourcemesh", filename.c_str());
	source.writeMesh(sourcefilename);
	std::string targetfilename = cv::format("%s.targetmesh", filename.c_str());
	target.writeMesh(targetfilename);
	getDeformationScore();
	fs << "score" << score;
	fs << "shapeMat" << shapeMat;
	fs.release();
	return 0;
}

/**
* @brief read ASAP from file (only mesh is read from file in this function now)
* @param std::string filename
* @return int: ErrorCode
*/
int ASAP::readASAPMesh(std::string filename) {
	cv::FileStorage fs(filename, cv::FileStorage::READ);
	fs["height"] >> height;
	fs["width"] >> width;
	fs["quadHeight"] >> quadHeight;
	fs["quadWidth"] >> quadWidth;
	fs["imgHeight"] >> imgHeight;
	fs["imgWidth"] >> imgWidth;
	std::string sourcefilename = cv::format("%s.sourcemesh", filename.c_str());
	source.readMesh(sourcefilename);
	std::string targetfilename = cv::format("%s.targetmesh", filename.c_str());
	target.readMesh(targetfilename);
	fs["score"] >> score;
	fs["shapeMat"] >> shapeMat;
	fs.release();
	return 0;
}

cv::Mat ASAP::getScore() {
	return score;
}
cv::Mat ASAP::getShapeMat() {
	return shapeMat;
}
cv::Size ASAP::getMeshSize() {
	return cv::Size(width, height);
}

cv::Mat ASAP::genMaskFromScore(cv::Mat score, float th) {
	cv::Mat mask = cv::Mat::zeros(imgHeight, imgWidth, CV_32F);
	for (int i = 0; i < score.rows; i++) {
		for (int j = 0; j < score.cols; j++) {
			if (score.at<float>(i, j) < th)
				continue;
			Quad q = source.getQuad(i + 1, j + 1);
			for (int ii = q.getVertex(0, 0).y; ii <= q.getVertex(1, 1).y; ii++) {
				for (int jj = q.getVertex(0, 0).x; jj <= q.getVertex(1, 1).x; jj++) {
					mask.at<float>(ii, jj) = 1;
				}
			}
		}
	}
	// smooth
	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(30, 30));
	cv::dilate(mask, mask, kernel);
	cv::Mat smoothK = cv::getGaussianKernel(30, 20, CV_32F);
	cv::Mat smoothK2D = smoothK * smoothK.t();
	cv::filter2D(mask, mask, -1, smoothK2D, cv::Point(-1, -1), 0.0, cv::BORDER_REPLICATE);
	return mask;
}