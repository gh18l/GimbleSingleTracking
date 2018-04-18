/**
@brief standalone implementation of deepflow in opencv
@author: Shane Yuan
@date: Apr 19, 2016
*/

#ifndef DEEP_FLOW_H
#define DEEP_FLOW_H

// #include "VariationalRefinement.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/optflow.hpp>

namespace cv {
	namespace optflow {
		// Deep flow class
		typedef struct struct_DeepflowPara {
			float sigma; // Gaussian smoothing parameter
			int minSize; // minimal dimension of an image in the pyramid
			float downscaleFactor; // scaling factor in the pyramid
			int fixedPointIterations; // during each level of the pyramid
			int sorIterations; // iterations of SOR
			float alpha; // smoothness assumption weight
			float delta; // color constancy weight
			float gamma; // gradient constancy weight
			float omega; // relaxation factor in SOR
			int layers;
			struct_DeepflowPara() {
				sigma = 0.6f;
				minSize = 25;
				downscaleFactor = 0.95f;
				fixedPointIterations = 5;
				sorIterations = 25;
				alpha = 1.0f;
				delta = 0.5f;
				gamma = 5.0f;
				omega = 1.6f;
				layers = 20;
			}
		} DeepflowPara, VariationalRefinePara;
	

	class DeepFlow : public DenseOpticalFlow {
		public:
			DeepFlow();

			void calc(InputArray I0, InputArray I1, InputOutputArray flow);
			void collectGarbage();

			void setSigma(float sigma) { this->sigma = sigma; }
			float getSigma() { return sigma; }
			void setMiniSize(int minSize) { this->minSize = minSize; }
			float getMiniSize() { return minSize; }
			void setDownscaleFactor(float downscaleFactor) { this->downscaleFactor = downscaleFactor; }
			float getDownscaleFactor() { return downscaleFactor; }
			void setSorIterations(int sorIterations) { this->sorIterations = sorIterations; }
			int getSorIterations() { return sorIterations; }
			void setAlpha(float alpha) { this->alpha = alpha; }
			float getAlpha() { return alpha; }
			void setDelta(float delta) { this->delta = delta; }
			float getDelta() { return delta; }
			void setGamma(float gamma) { this->gamma = gamma; }
			float getGamma() { return gamma; }
			void setOmega(float omega) { this->omega = omega; }
			float getOmega() { return omega; }
			void setDeepflowPara(DeepflowPara para) {
				sigma = para.sigma;
				minSize = para.minSize;
				downscaleFactor = para.downscaleFactor;
				fixedPointIterations = para.fixedPointIterations;
				sorIterations = para.sorIterations;
				alpha = para.alpha;
				delta = para.delta;
				gamma = para.gamma;
				omega = para.omega;
			}

		protected:
			float sigma; // Gaussian smoothing parameter
			int minSize; // minimal dimension of an image in the pyramid
			float downscaleFactor; // scaling factor in the pyramid
			int fixedPointIterations; // during each level of the pyramid
			int sorIterations; // iterations of SOR
			float alpha; // smoothness assumption weight
			float delta; // color constancy weight
			float gamma; // gradient constancy weight
			float omega; // relaxation factor in SOR

			int maxLayers; // max amount of layers in the pyramid
			int interpolationType;

		private:
			std::vector<Mat> buildPyramid(const Mat& src);

		};

		Ptr<DeepFlow> createDeepFlow();
	}
}

#endif