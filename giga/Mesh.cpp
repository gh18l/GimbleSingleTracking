/**
 * C++ source file of Quad class
 *
 * Author: Shane Yuan
 * Date: Jan 13, 2016
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
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

#include "Mesh.h"

Mesh::Mesh() {};

Mesh::Mesh(int rows, int cols, float quadWidth, float quadHeight) {
	this->imgRows = rows;
	this->imgCols = cols;
	this->quadWidth = quadWidth;
	this->quadHeight = quadHeight;

	vector<float> xSet;
	vector<float> ySet;

	float eps = 1;
	float x = 0;
	while (this->imgCols - x > 0.5 * quadWidth) {
		xSet.push_back(x);
		x += this->quadWidth;
	}
	xSet.push_back(this->imgCols - 1);

	float y = 0;
	while (this->imgRows - y > 0.5 * quadHeight) {
		ySet.push_back(y);
		y += this->quadHeight;
	}
	ySet.push_back(this->imgRows - 1);

	this->meshHeight = ySet.size();
	this->meshWidth = xSet.size();

	xMat.resize(this->meshHeight, this->meshWidth);
	yMat.resize(this->meshHeight, this->meshWidth);

	for (y = 0; y < this->meshHeight; y ++) {
		for (x = 0; x < this->meshWidth; x ++)
		{
			this->xMat(y, x) = xSet[x];
			this->yMat(y, x) = ySet[y];
		}
	}
}

Mesh::~Mesh() {};

// get vertex
Point2f Mesh::getVertex(int i, int j) {
	return Point2f(xMat(i, j), yMat(i, j));
}

// set vertex
int Mesh::setVertex(int i, int j, Point2f p) {
	xMat(i, j) = p.x;
	yMat(i, j) = p.y;
	return 0;
}

// get quad
Quad Mesh::getQuad(int i, int j) {
	Point2f v00 = getVertex(i - 1, j - 1);
	Point2f v01 = getVertex(i - 1, j);
	Point2f v10 = getVertex(i, j - 1);
	Point2f v11 = getVertex(i, j);
	return Quad(v00, v01, v10, v11);
}

// get function
int Mesh::getMeshHeight() {
	return this->meshHeight;
}
int Mesh::getMeshWidth() {
	return this->meshWidth;
}

int Mesh::calcDeformationScore() {
	meshHeight = getMeshHeight();
	meshWidth = getMeshWidth();
	deformationScore = cv::Mat::zeros(meshHeight - 1, meshWidth - 1, CV_32FC4);
	for (int i = 1; i < meshHeight; i ++) {
		for (int j = 1; j < meshWidth; j ++) {
			Quad q = getQuad(i, j);
			deformationScore.at<cv::Vec4f>(i - 1, j - 1) = q.getDeformationScore();
		}
	}
	return 0;
}

cv::Mat Mesh::getDeformationScore() {
	return deformationScore;
}

int Mesh::writeMesh(std::string filename) {
	std::fstream fs(filename, std::ios::out);
	fs << imgRows << std::endl;
	fs << imgCols << std::endl;
	fs << meshWidth << std::endl;
	fs << meshHeight << std::endl;
	fs << quadWidth << std::endl;
	fs << quadHeight << std::endl;

	for (int y = 0; y < this->meshHeight; y++) {
		for (int x = 0; x < this->meshWidth; x++) {
			fs << xMat(y, x) << '\t' << yMat(y, x) << std::endl;
		}
	}

	fs.close();
	return 0;
}

int Mesh::readMesh(std::string filename) {
	std::fstream fs(filename, std::ios::in);
	fs >> imgRows;
	fs >> imgCols;
	fs >> meshWidth;
	fs >> meshHeight;
	fs >> quadWidth;
	fs >> quadHeight;

	xMat.resize(this->meshHeight, this->meshWidth);
	yMat.resize(this->meshHeight, this->meshWidth);

	for (int y = 0; y < this->meshHeight; y++) {
		for (int x = 0; x < this->meshWidth; x++) {
			fs >> xMat(y, x);
			fs >> yMat(y, x);
		}
	}

	fs.close();
	return 0;
}