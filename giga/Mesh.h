/**
 * head file of Mesh class
 *
 * Author: Shane Yuan
 * Date: Jan 13, 2016
 *
 */

#ifndef VIDEOSTAB_MESH_H
#define VIDEOSTAB_MESH_H

#include "Quad.h"

class Mesh
{
	// varible
private:
	
public:
	int imgRows;
	int imgCols;
	int meshWidth;
	int meshHeight;
	float quadWidth;
	float quadHeight;
	Eigen::MatrixXf xMat;
	Eigen::MatrixXf yMat;
	cv::Mat deformationScore;

	// function
private:


public:
	Mesh();
	Mesh(int rows, int cols, float quadWidth, float quadHeight);
	Mesh& operator= (const Mesh& mesh) {
		if (this == &mesh)
			return *this;
		this->imgRows = mesh.imgRows;
		this->imgCols = mesh.imgCols;
		this->meshWidth = mesh.meshWidth;
		this->meshHeight = mesh.meshHeight;
		this->quadWidth = mesh.quadWidth;
		this->quadHeight = mesh.quadHeight;

		xMat.resize(meshHeight, meshWidth);
		yMat.resize(meshHeight, meshWidth);
		for (int i = 0; i < meshHeight; i ++) {
			for (int j = 0; j < meshWidth; j ++) {
				this->xMat(i, j) = mesh.xMat(i, j);
				this->yMat(i, j) = mesh.yMat(i, j);
			}
		}

		deformationScore = mesh.deformationScore.clone();

		return *this;
	}
	~Mesh();

	cv::Point2f getVertex(int i, int j);
	int setVertex(int i, int j, cv::Point2f p);
	Quad getQuad(int i, int j);

	int getMeshHeight();
	int getMeshWidth();

	int calcDeformationScore();
	cv::Mat getDeformationScore();

	int writeMesh(std::string filename);
	int readMesh(std::string filename);
};

#endif
