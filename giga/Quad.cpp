/**
 * C++ source file of Quad class
 *
 * Author: Shane Yuan
 * Date: Jan 12, 2016
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

// eigen
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace std;
using namespace cv;

#include "Quad.h"

Quad::Quad(Point2f v00, Point2f v01, Point2f v10, Point2f v11) {
	this->v00 = v00;
	this->v01 = v01;
	this->v10 = v10;
	this->v11 = v11;
}

Quad::~Quad() {};

// get minimum and maximum values
float Quad::getMinX() {
	return min(min(min(v00.x, v01.x), v10.x), v11.x);
}
float Quad::getMaxX() {
	return max(max(max(v00.x, v01.x), v10.x), v11.x);
}
float Quad::getMinY() {
	return min(min(min(v00.y, v01.y), v10.y), v11.y);
}
float Quad::getMaxY() {
	return max(max(max(v00.y, v01.y), v10.y), v11.y);
}

cv::Point2f Quad::getVertex(int i, int j) {
	if ((i == 0) && (j == 0))
		return v00;
	if ((i == 0) && (j == 1))
		return v01;
	if ((i == 1) && (j == 0))
		return v10;
	if ((i == 1) && (j == 1))
		return v11;
	else {
		std::cout << cv::format("Input parameter is wrong i = {0, 1}, j = {0, 1}, File: %s Line: %d", __FILE__, __LINE__) << std::endl;
		exit(-1);
	}
}

// check if point is in Triangle
bool Quad::isPointInTriangle(Point2f p, Point2f v0, Point2f v1, Point2f v2) {
	float lambda1;
	float lambda2;
	float lambda3;

	lambda1 = ((v1.y - v2.y) * (p.x - v2.x) + (v2.x - v1.x) * (p.y - v2.y)) /
		((v1.y - v2.y) * (v0.x - v2.x) + (v2.x - v1.x) * (v0.y - v2.y));
	lambda2 = ((v2.y - v0.y) * (p.x - v2.x) + (v0.x - v2.x) * (p.y - v2.y)) /
		((v2.y - v0.y) * (v1.x - v2.x) + (v0.x - v2.x) * (v1.y - v2.y));
	lambda3 = 1 - lambda1 - lambda2;

	if ((lambda1 >= 0.0) && (lambda1 <= 1.0) && (lambda2 >= 0.0) && (lambda2 <=1.0)
		&& (lambda3 >= 0.0) && (lambda3 <= 1.0)) {
		return true;
	}
	return false;
}

// check if point is in quad
bool Quad::isPointInQuad(Point2f p) {
	bool in1 = isPointInTriangle(p, this->v00, this->v01, this->v11);
	bool in2 = isPointInTriangle(p, this->v00, this->v10, this->v11);
	if ((in1 == true) || (in2 = true)) {
		return true;
	}
	return false;
}

BilinearCoeff Quad::getBilinearCoeff(Point2f p) {
	BilinearCoeff coef;

	float aX = v00.x - v01.x - v10.x + v11.x;
	float bX = -v00.x + v01.x;
	float cX = -v00.x + v10.x;
	float dX = v00.x - p.x;

	float aY = v00.y - v01.y - v10.y + v11.y;
	float bY = -v00.y + v01.y;
	float cY = -v00.y + v10.y;
	float dY = v00.y - p.y;

	float bigA = -aY*bX + bY*aX;
	float bigB = -aY*dX - cY*bX + dY*aX + bY*cX;
	float bigC = -cY*dX + dY*cX;

	float tmp1 = -1;
	float tmp2 = -1;
	float tmp3 = -1;
	float tmp4 = -1;

	float smallOne = 0.999999;
	float bigOne = 1.000001;
	float eps = 0.000001;

	float k1 = -1;
	float k2 = -1;

	if (bigB * bigB - 4 * bigA * bigC >= 0.0) {
		if (abs(bigA) >= eps) {
			tmp1 = (-bigB + sqrt(bigB * bigB - 4 * bigA * bigC)) / (2 * bigA);
			tmp2 = (-bigB - sqrt(bigB * bigB - 4 * bigA * bigC)) / (2 * bigA);
		}
		else {
			tmp1 = -bigC / bigB;
		}

		if ((tmp1 >= -smallOne) && (tmp1 <= bigOne)) {
			tmp3 = -(bY * tmp1 + dY) / (aY * tmp1 + cY);
			tmp4 = -(bX * tmp1 + dX) / (aX * tmp1 + cX);
			if ((tmp3 >= -smallOne) && (tmp3 <= bigOne)) {
				k1 = tmp1;
				k2 = tmp3;
			} 
			else if ((tmp4 >= -smallOne) && (tmp4 <= bigOne)){
				k1 = tmp1;
				k2 = tmp4;
			}
		}

		if ((tmp2 >= -smallOne) && (tmp2 <= bigOne)) {
			if ((tmp3 >= -smallOne) && (tmp3 <= bigOne)) {
				k1 = tmp2;
				k2 = tmp3;
			}
			else if ((tmp4 >= -smallOne) && (tmp4 <= bigOne)){
				k1 = tmp2;
				k2 = tmp4;
			}
		}
	}

	if ((k1 == -1)||(k2 == -1)) {
#ifdef MY_DEBUG
		MyUtility::dumpLog(string("Quad.cpp get bilinear coefficients error!"));
		MyUtility::dumpLog(p);
		MyUtility::dumpLog(v00);
		MyUtility::dumpLog(v01);
		MyUtility::dumpLog(v10);
		MyUtility::dumpLog(v11);
		MyUtility::dumpLogTime();
#endif
		cerr << "Error in calculating bilinear coefficient of " << p << ", K1 and K2 is not initiated yet." << endl;
		exit(0);
	}

	if ((k1 >= -smallOne) && (k1 <= bigOne) && (k2 >= -smallOne) && (k2 <= bigOne)) {
		float coe1 = (1.0 - k1) * (1.0 - k2);
		float coe2 = k1 * (1.0 - k2);
		float coe3 = (1.0 - k1) * k2;
		float coe4 = k1 * k2;

		coef.coeff(0, 0) = coe1;
		coef.coeff(1, 0) = coe2;
		coef.coeff(2, 0) = coe3;
		coef.coeff(3, 0) = coe4;
		coef.eff = true;
	}
	else {
		coef.eff = 0;
	}

	return coef;
}

float Quad::calcVectorAngle(cv::Point2f left, cv::Point2f center, cv::Point2f right) {
	float val;
	cv::Point2f leftVec = left - center;
	leftVec = leftVec / cv::norm(leftVec);
	cv::Point2f rightVec = right - center;
	rightVec = rightVec / cv::norm(rightVec);
	val = leftVec.x * rightVec.x + leftVec.y * rightVec.y;
	return val;
}

cv::Vec4f Quad::getDeformationScore() {
	// score 00
	float score00 = calcVectorAngle(v10, v00, v01);
	// score 00
	float score01 = calcVectorAngle(v00, v01, v11);
	// score 00
	float score11 = calcVectorAngle(v01, v11, v10);
	// score 00
	float score10 = calcVectorAngle(v11, v10, v00);
	return cv::Vec4f(score00, score01, score10, score11);
}