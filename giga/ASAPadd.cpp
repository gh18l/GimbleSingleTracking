/**
 * C++ source file of as similar as possible warping class
 * source file for add functions
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

// eigen
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace std;
using namespace cv;

#include "ASAP.h"

/**
 * get smooth weight of a point in a triangle
 * Input:
 * Point2f v1: first point of triangle
 * Point2f v2: second point of triangle
 * Point2f v3: third point of triangle
 * Return:
 * Point2f : weight
 */
Point2f ASAP::getSmoothWeight(Point2f v1, Point2f v2, Point2f v3) {
	v1.x++;
	v1.y++;
	v2.x++;
	v2.y++;
	v3.x++;
	v3.y++;

	float d1 = sqrt((v1.x - v2.x) * (v1.x - v2.x) + (v1.y - v2.y) * (v1.y - v2.y));
	float d3 = sqrt((v2.x - v3.x) * (v2.x - v3.x) + (v2.y - v3.y) * (v2.y - v3.y));

	Point2f v21 = Point2f(v1.x - v2.x, v1.y - v2.y);
	Point2f v23 = Point2f(v3.x - v2.x, v3.y - v2.y);

	float cosin = v21.x*v23.x + v21.y*v23.y;
	cosin = cosin / (d1*d3);

	float u_dis = cosin*d1;
	float u = u_dis / d3;

	float v_dis = sqrt(d1*d1 - u_dis*u_dis);
	float v = v_dis / d3;

	return Point2f(u, v);
}

/**
 * add a row to smooth coefficient matrix
 * Input
 * int i: row index of quad
 * int j: column index of quad
 * float weight: weight of quad
 * Return:
 * int: ErrorCode
 */
int ASAP::addCoefficientRow(float val1, float val2, float val3) {
	smoothConstraints(SCcount, 0) = val1;
	smoothConstraints(SCcount, 1) = val2;
	smoothConstraints(SCcount, 2) = val3;
	SCcount++;
#ifdef MY_DEBUG
	cout << "SCcount = " << SCcount << endl;
#endif
	return 0;
}

/**
 * add smooth coefficient for quad (i, j)
 * Input
 * int i: row index of quad
 * int j: column index of quad
 * float weight: weight of quad
 * int model: choose which triangle smooth constrain to compute
 * Return:
 * int: ErrorCode
 */
int ASAP::addCoefficient(int i, int j, float weight, int model) {
	Point2f v1;
	Point2f v2;
	Point2f v3;
	int coordv1;
	int coordv2;
	int coordv3;
	
	if (model == 1) {
		// 	v3		
		// 	_________
		// 	|		 |
		// 	|________|   
		//	v2		v1(i, j)
		v1 = this->source.getVertex(i, j);
		v2 = this->source.getVertex(i, j - 1);
		v3 = this->source.getVertex(i - 1, j - 1);
		coordv1 = i * this->width + j;
		coordv2 = i * this->width + j - 1;
		coordv3 = (i - 1) * this->width + j - 1;
	}
	else if (model == 2) {
		// 	v3		v2
		// 	_________
		// 	|		 |
		// 	|________|   
		//			v1(i, j)
		v1 = this->source.getVertex(i, j);
		v2 = this->source.getVertex(i - 1, j);
		v3 = this->source.getVertex(i - 1, j - 1);
		coordv1 = i * this->width + j;
		coordv2 = (i - 1) * this->width + j;
		coordv3 = (i - 1) * this->width + j - 1;
	}
	else if (model == 3) {
		// 	v2		v3
		// 	_________
		// 	|		 |
		// 	|________|   
		//	v1(i, j)
		v1 = this->source.getVertex(i, j);
		v2 = this->source.getVertex(i - 1, j);
		v3 = this->source.getVertex(i - 1, j + 1);
		coordv1 = i * this->width + j;
		coordv2 = (i - 1) * this->width + j;
		coordv3 = (i - 1) * this->width + j + 1;
	}
	else if (model == 4) {
		// 			v3
		// 	_________
		// 	|		 |
		// 	|________|   
		//	v1(i, j) v2
		v1 = this->source.getVertex(i, j);
		v2 = this->source.getVertex(i, j + 1);
		v3 = this->source.getVertex(i - 1, j + 1);
		coordv1 = i * this->width + j;
		coordv2 = i * this->width + j + 1;
		coordv3 = (i - 1) * this->width + j + 1;
	}
	else if (model == 5) {
		// v1(i, j)	 v2
		// 	_________
		// 	|		 |
		// 	|________|   
		//			 v3
		v1 = this->source.getVertex(i, j);
		v2 = this->source.getVertex(i, j + 1);
		v3 = this->source.getVertex(i + 1, j + 1);
		coordv1 = i * this->width + j;
		coordv2 = i * this->width + j + 1;
		coordv3 = (i + 1) * this->width + j + 1;
	}
	else if (model == 6) {
		// v1(i, j)	 
		// 	_________
		// 	|		 |
		// 	|________|   
		//	v2		 v3
		v1 = this->source.getVertex(i, j);
		v2 = this->source.getVertex(i + 1, j);
		v3 = this->source.getVertex(i + 1, j + 1);
		coordv1 = i * this->width + j;
		coordv2 = (i + 1) * this->width + j;
		coordv3 = (i + 1) * this->width + j + 1;
	}
	else if (model == 7) {
		//			v1(i, j)	 
		// 	_________
		// 	|		 |
		// 	|________|   
		//	v3		 v2
		v1 = this->source.getVertex(i, j);
		v2 = this->source.getVertex(i + 1, j);
		v3 = this->source.getVertex(i + 1, j - 1);
		coordv1 = i * this->width + j;
		coordv2 = (i + 1) * this->width + j;
		coordv3 = (i + 1) * this->width + j - 1;
	}
	else if (model == 8) {
		// v2		v1(i, j)	 
		// 	_________
		// 	|		 |
		// 	|________|   
		//	v3		 
		v1 = this->source.getVertex(i, j);
		v2 = this->source.getVertex(i, j - 1);
		v3 = this->source.getVertex(i + 1, j - 1);
		coordv1 = i * this->width + j;
		coordv2 = i * this->width + j - 1;
		coordv3 = (i + 1) * this->width + j - 1;
	}
	
	Point2f uv = this->getSmoothWeight(v1, v2, v3);
	float u = uv.x;
	float v = uv.y;
	
	if (model % 2 == 1) {
		this->addCoefficientRow(rowCount, getXIndex(coordv2), (1.0 - u) * weight);
		this->addCoefficientRow(rowCount, getXIndex(coordv3), u * weight);
		this->addCoefficientRow(rowCount, getYIndex(coordv2), v * weight);
		this->addCoefficientRow(rowCount, getYIndex(coordv3), (-1.0 * v) * weight);
		this->addCoefficientRow(rowCount, getXIndex(coordv1), (-1.0) * weight);
		this->rowCount++;

		this->addCoefficientRow(rowCount, getYIndex(coordv2), (1.0 - u) * weight);
		this->addCoefficientRow(rowCount, getYIndex(coordv3), u * weight);
		this->addCoefficientRow(rowCount, getXIndex(coordv3), v * weight);
		this->addCoefficientRow(rowCount, getXIndex(coordv2), (-1.0 * v) * weight);
		this->addCoefficientRow(rowCount, getYIndex(coordv1), (-1.0) * weight);
		this->rowCount++;
	}
	else {
		this->addCoefficientRow(rowCount, getXIndex(coordv2), (1.0 - u) * weight);
		this->addCoefficientRow(rowCount, getXIndex(coordv3), u * weight);
		this->addCoefficientRow(rowCount, getYIndex(coordv3), v * weight);
		this->addCoefficientRow(rowCount, getYIndex(coordv2), (-1.0 * v) * weight);
		this->addCoefficientRow(rowCount, getXIndex(coordv1), (-1.0) * weight);
		this->rowCount++;

		this->addCoefficientRow(rowCount, getYIndex(coordv2), (1.0 - u) * weight);
		this->addCoefficientRow(rowCount, getYIndex(coordv3), u * weight);
		this->addCoefficientRow(rowCount, getXIndex(coordv2), v * weight);
		this->addCoefficientRow(rowCount, getXIndex(coordv3), (-1.0 * v) * weight);
		this->addCoefficientRow(rowCount, getYIndex(coordv1), (-1.0) * weight);
		this->rowCount++;
	}
#ifdef MY_DEBUG
	cout << rowCount << endl;
#endif
	return 0;
}

// get x index
int ASAP::getXIndex(int index) {
	return index;
}
// get y index
int ASAP::getYIndex(int index) {
	return index + this->width * this->height;
}

