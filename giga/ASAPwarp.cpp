/**
 * C++ source file of as similar as possible warping class
 * warping image part
 *
 * Author: Shane Yuan
 * Date: Jan 16, 2016
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

/**
 * warp image via flow field
 * Input 
 * Mat input: input image
 * return
 * Mat: output warped image
 */
Mat ASAP::warp(Mat input) {
	Mat output;
	remap(input, output, flowfield, Mat(), INTER_CUBIC);
	return output;
}

/**
 * calculate warping flow field
 * Input
 * Mat input: input image
 * return
 * Mat: output warped image
 */
int ASAP::calcFlowField() {
	flowfield = Mat::zeros(this->imgHeight, this->imgWidth, CV_32FC2);
	for (int i = 0; i < height - 1; i ++) {
		for (int j = 0; j < width - 1; j ++) {
#ifdef MY_DEBUG
			printf("Calculate flowfield for quad (%d, %d) ...\n", i, j);
			if (i == 1 && j == 0)
				i = 1;
#endif
			// get 4 source points
			vector<Point2f> vs;
			vs.resize(4);
			vs[0] = source.getVertex(i, j);
			vs[1] = source.getVertex(i, j + 1);
			vs[2] = source.getVertex(i + 1, j);
			vs[3] = source.getVertex(i + 1, j + 1);
			// get 4 target points
			vector<Point2f> vt;
			vt.resize(4);
			vt[0] = target.getVertex(i, j);
			vt[1] = target.getVertex(i, j + 1);
			vt[2] = target.getVertex(i + 1, j);
			vt[3] = target.getVertex(i + 1, j + 1);
			// get homography matrix
			Mat H = getPerspectiveTransform(vs, vt);
			H.convertTo(H, CV_32F);
#ifdef MY_DEBUG
			MyUtility::dumpLog(cv::format("i = %d, j = %d", i, j));
			MyUtility::dumpLog(H);
			MyUtility::dumpLogTime();
#endif
			// calculate target points of all the pixels in this quad
			Mat pts = Mat::zeros(3, (static_cast<int>(vs[3].y) - static_cast<int>(vs[0].y) + 1) * (static_cast<int>(vs[3].x) - static_cast<int>(vs[0].x) + 1), CV_32F);
			int index = 0;
			for (int pi = static_cast<int>(vs[0].y); pi <= static_cast<int>(vs[3].y); pi++) {
				for (int pj = static_cast<int>(vs[0].x); pj <= static_cast<int>(vs[3].x); pj++) {
					pts.at<float>(0, index) = pj;
					pts.at<float>(1, index) = pi;
					pts.at<float>(2, index) = 1.0;
					index++;
				}
			}
			pts = H * pts;
			index = 0;
			for (int pi = static_cast<int>(vs[0].y); pi <= static_cast<int>(vs[3].y); pi++) {
				for (int pj = static_cast<int>(vs[0].x); pj <= (vs[3].x); pj++) {
					Point2f p;
					p.x = pts.at<float>(0, index) / pts.at<float>(2, index);
					p.y = pts.at<float>(1, index) / pts.at<float>(2, index);
					flowfield.at<Point2f>(pi, pj) = p;
					index++;
				}
			}
		}
	}
	// smooth
	cv::Mat smoothK = cv::getGaussianKernel(5, 2, CV_32F);
	cv::Mat smoothK2D = smoothK * smoothK.t();
	cv::filter2D(flowfield, flowfield, -1, smoothK2D, cv::Point(-1, -1), 0.0, cv::BORDER_REPLICATE);
	return 0;
}



/**
* draw mesh on images for visualization
* @param cv::Mat input: input image
* @param int func: 0, draw source, 1, draw target, 2, draw source and deformation, 3 draw target and deformation
* @param cv::Mat & output: output imge with drawed mesh
* @param cv::Scalar dotColor: color of dot 
* @param cv::Scalar lineColor: color of line
* @param int dotRadius: radius of drawn dot
* @param int lineWidth: width of drawn line
* @param int isDeformation: 1: draw deformation, 0: not draw deformation
* @param float th: use the complementary color to draw the mesh quads whose score is small than th
*/
int ASAP::drawMesh(cv::Mat input, int func, cv::Mat & output, cv::Scalar dotColor, cv::Scalar lineColor, int dotRadius, int lineWidth, float th /* = 0.25*/) {
	getDeformationScore();
	output = input.clone();
	if ((func == 0) || (func == 2)) {
		for (int i = 0; i < source.getMeshHeight(); i++) {
			for (int j = 0; j < source.getMeshWidth(); j++) {
				cv::Point2f v00 = source.getVertex(i, j);
				cv::circle(output, v00, dotRadius, dotColor, -1, 8, 0);
				if (i < source.getMeshHeight() - 1) {
					cv::Point2f v10 = source.getVertex(i + 1, j);
					cv::circle(output, v10, dotRadius, dotColor, -1, 8, 0);
					cv::line(output, v00, v10, lineColor, lineWidth, cv::LINE_8, 0);
				}
				if (j < source.getMeshWidth() - 1) {
					cv::Point2f v01 = source.getVertex(i, j + 1);
					cv::circle(output, v01, dotRadius, dotColor, -1, 8, 0);
					cv::line(output, v00, v01, lineColor, lineWidth, cv::LINE_8, 0);
				}
			}
		}
		if (func == 2) {
			cv::Scalar dotColor2 = cv::Scalar(dotColor.val[2], dotColor.val[1], dotColor.val[0]);
			cv::Scalar lineColor2 = cv::Scalar(lineColor.val[2], lineColor.val[1], lineColor.val[0]);
			for (int i = 0; i < score.rows; i++) {
				for (int j = 0; j < score.cols; j++) {
					if (score.at<float>(i, j) < th)
						continue;
					Quad q = source.getQuad(i + 1, j + 1);
					cv::circle(output, q.getVertex(0, 0), dotRadius, dotColor2, -1, 8, 0);
					cv::circle(output, q.getVertex(0, 1), dotRadius, dotColor2, -1, 8, 0);
					cv::circle(output, q.getVertex(1, 0), dotRadius, dotColor2, -1, 8, 0);
					cv::circle(output, q.getVertex(1, 1), dotRadius, dotColor2, -1, 8, 0);
					cv::line(output, q.getVertex(0, 0), q.getVertex(0, 1), lineColor2, lineWidth, cv::LINE_8, 0);
					cv::line(output, q.getVertex(0, 0), q.getVertex(1, 0), lineColor2, lineWidth, cv::LINE_8, 0);
					cv::line(output, q.getVertex(1, 1), q.getVertex(0, 1), lineColor2, lineWidth, cv::LINE_8, 0);
					cv::line(output, q.getVertex(1, 1), q.getVertex(1, 0), lineColor2, lineWidth, cv::LINE_8, 0);
				}
			}
		}
		
	}
	else if ((func == 1) || (func == 3)) {
		for (int i = 0; i < target.getMeshHeight(); i++) {
			for (int j = 0; j < target.getMeshWidth(); j++) {
				cv::Point2f v00 = target.getVertex(i, j);
				cv::circle(output, v00, dotRadius, dotColor, -1, 8, 0);
				if (i < target.getMeshHeight() - 1) {
					cv::Point2f v10 = target.getVertex(i + 1, j);
					cv::circle(output, v10, dotRadius, dotColor, -1, 8, 0);
					cv::line(output, v00, v10, lineColor, lineWidth, cv::LINE_8, 0);
				}
				if (j < target.getMeshWidth() - 1) {
					cv::Point2f v01 = target.getVertex(i, j + 1);
					cv::circle(output, v01, dotRadius, dotColor, -1, 8, 0);
					cv::line(output, v00, v01, lineColor, lineWidth, cv::LINE_8, 0);
				}
			}
		}
		if (func == 3) {
			cv::Scalar dotColor2 = cv::Scalar(dotColor.val[2], dotColor.val[1], dotColor.val[0]);
			cv::Scalar lineColor2 = cv::Scalar(lineColor.val[2], lineColor.val[1], lineColor.val[0]);
			for (int i = 0; i < score.rows; i++) {
				for (int j = 0; j < score.cols; j++) {
					if (score.at<float>(i, j) < th)
						continue;
					Quad q = target.getQuad(i + 1, j + 1);
					cv::circle(output, q.getVertex(0, 0), dotRadius, dotColor2, -1, 8, 0);
					cv::circle(output, q.getVertex(0, 1), dotRadius, dotColor2, -1, 8, 0);
					cv::circle(output, q.getVertex(1, 0), dotRadius, dotColor2, -1, 8, 0);
					cv::circle(output, q.getVertex(1, 1), dotRadius, dotColor2, -1, 8, 0);
					cv::line(output, q.getVertex(0, 0), q.getVertex(0, 1), lineColor2, lineWidth, cv::LINE_8, 0);
					cv::line(output, q.getVertex(0, 0), q.getVertex(1, 0), lineColor2, lineWidth, cv::LINE_8, 0);
					cv::line(output, q.getVertex(1, 1), q.getVertex(0, 1), lineColor2, lineWidth, cv::LINE_8, 0);
					cv::line(output, q.getVertex(1, 1), q.getVertex(1, 0), lineColor2, lineWidth, cv::LINE_8, 0);
				}
			}
		}
	}
	return 0;
}

/**
* draw mesh on images for visualization with padding
* @param cv::Mat input: input image
* @param int func: 0, draw source, 1, draw target, 2, draw source and deformation, 3 draw target and deformation
* @param cv::Mat & output: output imge with drawed mesh
* @param cv::Scalar dotColor: color of dot
* @param cv::Scalar lineColor: color of line
* @param int dotRadius: radius of drawn dot
* @param int lineWidth: width of drawn line
* @param int isDeformation: 1: draw deformation, 0: not draw deformation
* @param cv::Point2f padding: add padding into input
*/
int ASAP::drawMeshWithPadding(cv::Mat input, int func, cv::Mat & output, cv::Scalar dotColor, cv::Scalar lineColor, int dotRadius, int lineWidth, cv::Point2f padding) {
	int width = input.cols + 2 * padding.x;
	int height = input.rows + 2 * padding.y;
	// output = cv::Mat::Mat(height, width, CV_8UC3, cv::Scalar(255, 255, 255));
	output.create(height, width, CV_8UC3);
	output = cv::Scalar(255, 255, 255);
	cv::Rect rect;
	rect.x = padding.x;
	rect.y = padding.y;
	rect.width = input.cols;
	rect.height = input.rows;
	input.copyTo(output(rect));

	if ((func == 0) || (func == 2)) {
		for (int i = 0; i < source.getMeshHeight(); i++) {
			for (int j = 0; j < source.getMeshWidth(); j++) {
				cv::Point2f v00 = source.getVertex(i, j) + padding;
				cv::circle(output, v00, dotRadius, dotColor, -1, 8, 0);
				if (i < source.getMeshHeight() - 1) {
					cv::Point2f v10 = source.getVertex(i + 1, j) + padding;
					cv::circle(output, v10, dotRadius, dotColor, -1, 8, 0);
					cv::line(output, v00, v10, lineColor, lineWidth, cv::LINE_8, 0);
				}
				if (j < source.getMeshWidth() - 1) {
					cv::Point2f v01 = source.getVertex(i, j + 1) + padding;
					cv::circle(output, v01, dotRadius, dotColor, -1, 8, 0);
					cv::line(output, v00, v01, lineColor, lineWidth, cv::LINE_8, 0);
				}
			}
		}
	}
	else if ((func == 1) || (func == 3)) {
		for (int i = 0; i < target.getMeshHeight(); i++) {
			for (int j = 0; j < target.getMeshWidth(); j++) {
				cv::Point2f v00 = target.getVertex(i, j) + padding;
				cv::circle(output, v00, dotRadius, dotColor, -1, 8, 0);
				if (i < target.getMeshHeight() - 1) {
					cv::Point2f v10 = target.getVertex(i + 1, j) + padding;
					cv::circle(output, v10, dotRadius, dotColor, -1, 8, 0);
					cv::line(output, v00, v10, lineColor, lineWidth, cv::LINE_8, 0);
				}
				if (j < target.getMeshWidth() - 1) {
					cv::Point2f v01 = target.getVertex(i, j + 1) + padding;
					cv::circle(output, v01, dotRadius, dotColor, -1, 8, 0);
					cv::line(output, v00, v01, lineColor, lineWidth, cv::LINE_8, 0);
				}
			}
		}
	}

	return 0;
}