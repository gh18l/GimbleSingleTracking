/**
 * head file of as similar as possible warping
 *
 * Author: Shane Yuan
 * Date: Jan 13, 2016
 *
 */

#ifndef VIDEOSTAB_ASAP_H
#define VIDEOSTAB_ASAP_H

// eigen
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/SparseQR>
#include <Eigen/OrderingMethods>

#include "Mesh.h"

class ASAP
{
public:
	// basic parameter
	// mesh size
	int height;
	int width;
	// quad size
	float quadWidth;
	float quadHeight;
	// image size
	int imgHeight;
	int imgWidth;

	// Mesh
	Mesh source;
	Mesh target;

	// control points
	Eigen::MatrixXf sourcePts;
	Eigen::MatrixXf targetPts;

	// Data term constrains
	std::vector<int> dataI;
	std::vector<int> dataJ;
	std::vector<cv::Point2f> orgPt;
	std::vector<cv::Point2f> desPt;
	std::vector<float> ptWeight;
	std::vector<float> dataV00;
	std::vector<float> dataV01;
	std::vector<float> dataV10;
	std::vector<float> dataV11;
	Eigen::MatrixXf dataConstraints;
	int numDataCons;
	int DCcount;

	// Smooth term constrains
	int numSmoothCons;
	Eigen::MatrixXf smoothConstraints;
	int SCcount;

	// counter
	int rowCount;
	int columns;

	// warping
	cv::Mat flowfield;

	// deformation score
	cv::Mat score;
	cv::Mat shapeMat;

public:

protected:
	// compute smooth constrain
	int createSmoothCons(float weight);
	cv::Point2f getSmoothWeight(cv::Point2f v1, cv::Point2f v2, cv::Point2f v3);
	int addCoefficient(int i, int j, float weight, int model);
	int addCoefficientRow(float val1, float val2, float val3);

	// compute data constrain
	int createDataCons(Eigen::MatrixXf & b);

	// get index
	int getXIndex(int index);
	int getYIndex(int index);
public:
	ASAP();
	ASAP(int height, int width, float quadHeight, float quadWidth, float weight);
	~ASAP();
	// set mesh
	int setMesh(int height, int width, float quadHeight, float quadWidth, float weight);

	// set control points
	int setControlPoints(std::vector<cv::Point2f> inputsPts, std::vector<cv::Point2f> outputsPts, std::vector<float> ptWeight = std::vector<float>());
	int fusionCorrespondence(std::vector<cv::Point2f> detailPts, std::vector<cv::Point2f> refPts,  cv::Mat featurePointFlow, cv::Mat pixelValueFlow,
		std::vector<cv::Point2f> & outDetailPts, std::vector<cv::Point2f> & outRefPts, std::vector<float> & outPtWeights);
	
	// solve optimization
	int solve();

	// warp image via mesh
	int calcFlowField();
	cv::Mat warp(cv::Mat input);

	// draw mesh on image
	int drawMesh(cv::Mat input, int func, cv::Mat & output, cv::Scalar dotColor, cv::Scalar lineColor, int dotRadius, int lineWidth, float th = 0.25);
	int drawMeshWithPadding(cv::Mat input, int func, cv::Mat & output, cv::Scalar dotColor, cv::Scalar lineColor, int dotRadius, int lineWidth, cv::Point2f	padding);

	// get flow field
	cv::Mat getFlowfield();

	// get deformation score
	cv::Mat getDeformationScore();
	// get deformation mask
	int getDeformationMask(cv::Mat & mask, cv::Mat & score, float th = 0.25);
	// get score
	cv::Mat getScore();
	cv::Mat getShapeMat();
	// get MeshSize
	cv::Size getMeshSize();
	// generate mask from score
	cv::Mat genMaskFromScore(cv::Mat score, float th);

	// Mesh IO function
	int writeASAPMesh(std::string filename);
	int readASAPMesh(std::string filename);
};

#endif

