/**
@brief standalone implementation of deepflow in opencv
@author: Shane Yuan
@date: Apr 19, 2016
*/

#include "DeepFlow.h"

namespace cv {
	namespace optflow {
		DeepFlow::DeepFlow()
		{
			// parameters
			sigma = 0.6f;
			minSize = 25;
			downscaleFactor = 0.95f;
			fixedPointIterations = 5;
			sorIterations = 25;
			alpha = 1.0f;
			delta = 0.5f;
			gamma = 5.0f;
			omega = 1.6f;

			//consts
			interpolationType = INTER_LINEAR;
			maxLayers = 200;
		}

		std::vector<Mat> DeepFlow::buildPyramid(const Mat& src)
		{
			std::vector<Mat> pyramid;
			pyramid.push_back(src);
			Mat prev = pyramid[0];
			int i = 0;
			while (i < this->maxLayers)
			{
				Mat next; //TODO: filtering at each level?
				Size nextSize((int)(prev.cols * downscaleFactor + 0.5f),
					(int)(prev.rows * downscaleFactor + 0.5f));
				if (nextSize.height <= minSize || nextSize.width <= minSize)
					break;
				resize(prev, next,
					nextSize, 0, 0,
					interpolationType);
				pyramid.push_back(next);
				prev = next;
			}
			return pyramid;
		}

		void DeepFlow::calc(InputArray _I0, InputArray _I1, InputOutputArray _flow)
		{
			Mat I0temp = _I0.getMat();
			Mat I1temp = _I1.getMat();

			CV_Assert(I0temp.size() == I1temp.size());
			CV_Assert(I0temp.type() == I1temp.type());
			CV_Assert(I0temp.channels() == 1);
			// TODO: currently only grayscale - data term could be computed in color version as well...

			Mat I0, I1;

			I0temp.convertTo(I0, CV_32F);
			I1temp.convertTo(I1, CV_32F);

			if (_flow.getMat().rows > 0) {
				resize(_flow, _flow, I0.size());
			}
			else {
				_flow.create(I0.size(), CV_32FC2);
			}
			Mat W = _flow.getMat(); // if any data present - will be discarded

			// pre-smooth images
			int kernelLen = ((int)floor(3 * sigma) * 2) + 1;
			Size kernelSize(kernelLen, kernelLen);
			GaussianBlur(I0, I0, kernelSize, sigma);
			GaussianBlur(I1, I1, kernelSize, sigma);
			// build down-sized pyramids
			std::vector<Mat> pyramid_I0 = buildPyramid(I0);
			std::vector<Mat> pyramid_I1 = buildPyramid(I1);
			int levelCount = (int)pyramid_I0.size();

			// initialize the first version of flow estimate to zeros
			Size smallestSize = pyramid_I0[levelCount - 1].size();
			W = Mat::zeros(smallestSize, CV_32FC2);

			for (int level = levelCount - 1; level >= 0; --level)
			{ //iterate through  all levels, beginning with the most coarse
				Ptr<VariationalRefinement> var = createVariationalFlowRefinement();

				var->setAlpha(4 * alpha);
				var->setDelta(delta / 3);
				var->setGamma(gamma / 3);
				var->setFixedPointIterations(fixedPointIterations);
				var->setSorIterations(sorIterations);
				var->setOmega(omega);

				var->calc(pyramid_I0[level], pyramid_I1[level], W);
				if (level > 0) //not the last level
				{
					Mat temp;
					Size newSize = pyramid_I0[level - 1].size();
					resize(W, temp, newSize, 0, 0, interpolationType); //resize calculated flow
					W = temp * (1.0f / downscaleFactor); //scale values
				}
			}
			W.copyTo(_flow);
		}

		void DeepFlow::collectGarbage() {}

		Ptr<DeepFlow> createDeepFlow() {
			return makePtr<DeepFlow>();
		}
	}
}