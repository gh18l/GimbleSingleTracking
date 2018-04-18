/**
 * head file of Quad class
 *
 * Author: Shane Yuan
 * Date: Jan 12, 2016
 *
 */

#ifndef VIDEOSTAB_QUAD_H
#define VIDEOSTAB_QUAD_H

#include <stdlib.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

struct BilinearCoeff {
	Eigen::Vector4f coeff;
	bool eff;
};

class Quad
{
	// varible
private:
	cv::Point2f v00;
	cv::Point2f v01;
	cv::Point2f v10;
	cv::Point2f v11;

public:

	// function
private:
	bool isPointInTriangle(cv::Point2f p, cv::Point2f v0, cv::Point2f v1, cv::Point2f v2);
	bool isPointInQuad(cv::Point2f p);

	float calcVectorAngle(cv::Point2f left, cv::Point2f center, cv::Point2f right);

public:
	Quad(cv::Point2f v00, cv::Point2f v01, cv::Point2f v10, cv::Point2f v11);
	~Quad();

	// get bilinear interpolation coefficients
	BilinearCoeff getBilinearCoeff(cv::Point2f p);
	// get deformation score
	cv::Vec4f getDeformationScore();

	// get minimum and maximum
	float getMinX();
	float getMaxX();
	float getMinY();
	float getMaxY();

	// get vertex
	cv::Point2f getVertex(int i, int j);

};

#endif
