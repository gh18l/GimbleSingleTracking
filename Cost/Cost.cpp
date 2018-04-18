#include "Cost.h"
#include<opencv2\opencv.hpp>
#include<opencv2\optflow.hpp>

#include "opencv2/core.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"

#include <stdio.h>

#include "GenCameraDriver.h"
using namespace std;
using namespace cv;
extern std::vector<cam::GenCamInfo> camInfos;
extern std::shared_ptr<cam::GenCamera> cameraPtr;
#define _DEBUG_COST     //注释掉为release模式
//#define VISUALIZE

Cost::Cost() 
{
	bool HOG = true;
	bool FIXEDWINDOW = false;
	bool MULTISCALE = false;
	bool SILENT = false;
	bool LAB = false;

	KCF_tracker.KCFTrackerinit(HOG, FIXEDWINDOW, MULTISCALE, LAB);
	isfind_max = 100;
	isfind_vec.resize(isfind_max);
	current_show.resize(2);   //只显示行人
}
Cost::~Cost() {}


cv::Mat Cost::cal_dis(cv::Mat flow_uv0, cv::Mat flow_uv1)
{
	cv::Mat out(flow_uv0.rows, flow_uv0.cols, flow_uv0.type());
	for (int i = 0; i < flow_uv0.rows; i++)
	{
		float* data1 = flow_uv0.ptr<float>(i);
		float* data2 = flow_uv1.ptr<float>(i);
		float* data3 = out.ptr<float>(i);
		for (int j = 0; j < flow_uv0.cols*3; j++)
		{
			/*if (data1[j] * data1[j] + data2[j] * data2[j] > 0.5)
			continue;*/
			data3[j] = data1[j] * data1[j] + data2[j] * data2[j];
		}
	}
	return out;
}

int Cost::flow(cv::Mat src1, cv::Mat src2, cv::Point &point)
{
	cv::Mat roi1, roi2;

	cv::Mat flow, flow_uv[2];
	cv::Mat move;   ////optical flow image
	cv::Mat out = cv::Mat::zeros(src1.rows, src1.cols, CV_8UC1);

	cv::Ptr<cv::DenseOpticalFlow> algorithm = cv::optflow::createOptFlow_SparseToDense();
	int max = 0;
	cv::Point maxpoint;

	for (int i = 0; ; i++)
	{
		cv::Point maxLoc;
		double max_temp;
		cv::cvtColor(roi1, roi1, CV_BGR2GRAY);
		cv::cvtColor(roi2, roi2, CV_BGR2GRAY);
		algorithm->calc(roi1, roi2, flow);
		split(flow, flow_uv);
		move=cal_dis(flow_uv[0], flow_uv[1]);
		cv::minMaxLoc(move, NULL, &max_temp, NULL, &maxLoc);
		if (max_temp > max)
		{
			max = max_temp;
			maxpoint = maxLoc;
		}
		//dis.convertTo(dis, CV_8U);   
	}
	point = maxpoint;
	return 0;
}

int Cost::video_initflow(cv::Mat src)
{	
	fre = cv::Mat::zeros(src.rows, src.cols, CV_32S);
	return 0;
}

int Cost::video_preflow(cv::Mat src)
{
	cv::Mat fgmask;
	bg_model->apply(src, fgmask, update_bg_model ? -1 : 0);
	return 0;
}

int Cost::video_updateflow(cv::Mat src, cv::Point& point)
{
	/*for (int i = 0; i < imgs.size(); i++)
	{
	resize(imgs[i], imgs[i], Size(1000,750), 0, 0, INTER_LINEAR_EXACT);
	}*/
	clock_t start, finish;
	cv::Mat fgmask,temp;
	cv::Mat dst;
	bg_model->apply(src, fgmask, update_bg_model ? -1 : 0);
	if (smoothMask)
	{
		GaussianBlur(fgmask, fgmask, Size(11, 11), 3.5, 3.5);
		threshold(fgmask, fgmask, 10, 255, THRESH_BINARY);
	}
	cv::Mat ker = cv::Mat::ones(src.rows*0.12, src.cols*0.12, CV_32F);
	filter2D(fgmask, dst, CV_32F, ker);    //卷积一次需要50
	//////增加cost项：云台转动距离，单通道，不在附近区域，就讲数值除以2，这样就不会不断跳动了
	for (int i = 0; i < dst.rows; i++)
	{
		int* data = dst.ptr<int>(i);
		for (int j = 0; j < dst.cols; j++)
		{
			//data[j] = (point.x - j)*(point.x - j) + (point.y - i)*(point.y - i);     //最大可能是6250000，一般可能也就是500*500=250000
			if (abs(point.x - j) > 200 || abs(point.y - i) > 150)
			{
				data[j] = data[j];
			}
		}
	}
	cv::Point maxLoc;
	minMaxLoc(dst, NULL, NULL, NULL, &maxLoc);

	fgmask.convertTo(fgmask, CV_8U);
	/////加个上限///////
	if (maxLoc.x - src.cols*0.06 < 0)
	{
		maxLoc.x = src.cols*0.06;
	}
	if (maxLoc.y - src.rows*0.06 < 0)
	{
		maxLoc.y = src.rows*0.06;
	}
	if (maxLoc.x + src.cols*0.06 >= dst.cols)
	{
		maxLoc.x = dst.cols- src.cols*0.06-1;
	}
	if (maxLoc.y + src.rows*0.06 >= dst.rows)
	{
		maxLoc.y = dst.rows - src.rows*0.06 - 1;
	}
	cv::rectangle(fgmask, cv::Point(maxLoc.x - src.cols*0.06, maxLoc.y - src.rows*0.06), cv::Point(maxLoc.x + src.cols*0.06, maxLoc.y + src.rows*0.06), cv::Scalar(255, 255, 255),10);
	fgmask.copyTo(temp);
	resize(temp, temp, Size(800, 600));
	imshow("mask", temp);
	cv::waitKey(30);
	//int *data = fre.ptr<int>(maxLoc.y);
	//data[maxLoc.x] += 1;

	//while (1)    //判断这个点是否被遍历超过10次
	//{
	//	int *data = fre.ptr<int>(maxLoc.y);
	//	if (data[maxLoc.x] > fre_tre)    //这个点不能要了
	//	{
	//		int *data = dst.ptr<int>(maxLoc.y);
	//		data[maxLoc.x] =0;
	//		minMaxLoc(dst, NULL, NULL, NULL, &maxLoc);
	//	}
	//	else   //找到一个可以用的点
	//	{
	//		break;
	//	}
	//}


	
	
	if (abs(maxLoc.x - src.cols*0.06 - point.x) < 50 && abs(maxLoc.y - src.rows*0.06 - point.y) < 35)     //离得太近了，云台不动，但计数器计数了
		return -1;

	point.x = maxLoc.x - int(src.cols*0.06);
	point.y = maxLoc.y - int(src.rows*0.06);
	//std::cout <<"I want to go "<< maxLoc << std::endl;
	return 0;
}

int Cost::init_people_detection(cv::Mat img)
{
	std::string cfgfile = "E:/data/YOLO/yolo.cfg";
	std::string weightfile = "E:/data/YOLO/yolo.weights";
	detector->init(cfgfile, weightfile, 0);
	SetBlock(img);
	return 0;
}

////////set image to m rows and n cols
////////from left to right save
int Cost::SetBlock(cv::Mat img)
{
	for (int i = 0; i < rows_b; i++)
	{
		for (int j = 0; j < cols_b; j++)
		{
			cv::Rect rect(j * img.cols / cols_b, i * img.rows / rows_b,
				img.cols / cols_b, img.rows / rows_b);
			img_rect.push_back(rect);
		}
	}
	return 0;
}

///////m rows and n cols, choose x blocks into net
int Cost::people_detection(cv::Mat& img)
{
	cv::Mat fgmask;
	bg_model->apply(img, fgmask, update_bg_model ? -1 : 0);

	if (smoothMask) 
	{
		GaussianBlur(fgmask, fgmask, Size(11, 11), 3.5, 3.5);
		threshold(fgmask, fgmask, 10, 255, THRESH_BINARY);
	}

	float ratio_width;
	float ratio_height;
	std::vector<int> sum;
	for (int i = 0; i < img_rect.size(); i++)
	{
		sum.push_back(GetSum(fgmask(img_rect[i])));
	}
	std::vector<int> index;
	index = choose_maxs(sum);
	std::vector<bbox_t> result_vec;
	for(int b = 0;b < set_b;b++)
	{
		cv::Mat img_region;
		img(img_rect[index[b]]).copyTo(img_region);
		ratio_width = static_cast<float>(img_region.cols) / 416.0f;
		ratio_height = static_cast<float>(img_region.rows) / 416.0f;
		cv::resize(img_region, img_region, cv::Size(416, 416));

		cv::Mat imgf;
		cv::cvtColor(img_region, imgf, cv::COLOR_BGR2RGB);
		imgf.convertTo(imgf, CV_32F, 1.0 / 255.0);

		float *img_h = new float[imgf.rows * imgf.cols * 3];
		size_t count = 0;
		for (size_t k = 0; k < 3; ++k) {
			for (size_t i = 0; i < imgf.rows; ++i) {
				for (size_t j = 0; j < imgf.cols; ++j) {
					img_h[count++] = imgf.at<cv::Vec3f>(i, j).val[k];
				}
			}
		}

		float* img_d;
		cudaMalloc(&img_d, sizeof(float) * 3 * img_region.rows * img_region.cols);
		cudaMemcpy(img_d, img_h, sizeof(float) * 3 * img_region.rows * img_region.cols,
			cudaMemcpyHostToDevice);


		detector->detect(img_d, img_region.cols, img_region.rows, result_vec);
		for (size_t i = 0; i < result_vec.size(); i++) {
			result_vec[i].x *= ratio_width;
			result_vec[i].y *= ratio_height;
			result_vec[i].w *= ratio_width;
			result_vec[i].h *= ratio_height;

			result_vec[i].x += img_rect[index[b]].x;
			result_vec[i].y += img_rect[index[b]].y;
			if(result_vec[i].prob > 0.25 && result_vec[i].obj_id == 0 && result_vec[i].w*result_vec[i].h<15000)
				detect_vec.push_back(result_vec[i]);
		}
#ifdef VISUALIZE
		for (size_t i = 0; i < result_vec.size(); i++)
		{
			if (result_vec[i].prob < 0.25 || result_vec[i].obj_id != 0 || result_vec[i].w*result_vec[i].h>15000)
			{
				continue;
			}
			cv::rectangle(img, cv::Rect(result_vec[i].x, result_vec[i].y, result_vec[i].w, result_vec[i].h),
				cv::Scalar(255, 0, 0), 5);
		}
		cv::rectangle(img, img_rect[index[b]],
			cv::Scalar(0, 255, 0), 5);

#endif
		result_vec.clear();
		cudaFree(img_d);
	}
	return 0;
}

int Cost::GetSum(cv::Mat input)
{
	size_t sum = 0;
	for (int i = 0; i < input.rows; i++)
	{
		uchar* data = input.ptr<uchar>(i);
		for (int j = 0; j < input.cols; j++)     //这里一定先检验一下fgmask是几个通道
		{
			sum += data[j];
		}
	}

	return sum;
}

//// choose x maxs from vector
std::vector<int> Cost::choose_maxs(std::vector<int> sum)
{
	std::vector<int>index(set_b);
	for (int i = 0; i < set_b; i++)
	{
		index[i] = i;
	}
	for (int i = set_b; i < sum.size(); i++)
	{
		int min_index = find_min(index, sum);
		if (sum[i] > sum[index[min_index]])
		{
			index[min_index] = i;
		}
	}
	return index;
}

//////find a min from specified index vector, it is number in index vector
int Cost::find_min(std::vector<int> index, std::vector<int>sum)
{
	size_t min = 999999999;
	int min_index;
	for (int i = 0; i < index.size(); i++)
	{
		if (sum[index[i]] < min)
		{
			min = sum[index[i]];
			min_index = i;
		}
	}
	return min_index;
}

int Cost::video_updatedetection(cv::Mat src, cv::Point src_point,cv::Point& dst_point)
{
	people_detection(src);

	//没检测到人
	if (detect_vec.size() == 0)
	{
		return -1;
	}
	cv::Mat src_rec;
	src.copyTo(src_rec);
	for (int i = 0; i < detect_vec.size(); i++)
	{
		cv::rectangle(src_rec, cv::Rect(detect_vec[i].x, detect_vec[i].y,
			detect_vec[i].w, detect_vec[i].h),
				cv::Scalar(255, 0, 0), 5);
	}

	for (int i = 0; i < detect_vec.size(); i++)
		current_vec = detect_vec[0];
		
	src_rec.copyTo(show_opencv);
	//int *data = fre.ptr<int>(maxLoc.y);
	//data[maxLoc.x] += 1;

	//while (1)    //判断这个点是否被遍历超过10次
	//{
	//	int *data = fre.ptr<int>(maxLoc.y);
	//	if (data[maxLoc.x] > fre_tre)    //这个点不能要了
	//	{
	//		int *data = dst.ptr<int>(maxLoc.y);
	//		data[maxLoc.x] =0;
	//		minMaxLoc(dst, NULL, NULL, NULL, &maxLoc);
	//	}
	//	else   //找到一个可以用的点
	//	{
	//		break;
	//	}
	//}




	//if (abs(current_vec.x - src.cols*0.06 - src_point.x) < 50 && abs(current_vec.y - src.rows*0.06 - src_point.y) < 35)     //离得太近了，云台不动，但计数器计数了
	//{
	//	detect_vec.clear();
	//	return -1;
	//}

	dst_point.x = current_vec.x - static_cast<int>(src.cols*0.06);
	dst_point.y = current_vec.y - static_cast<int>(src.rows*0.06);
	//std::cout << "I want to go " << current_vec.x <<"   "<< current_vec.y<< std::endl;
	detect_vec.clear();
	return 0;
}

//int Cost::trackingSetBlock(cv::Mat frame)
//{
//	scale = 200.0f / static_cast<float>(tracking_roi.height);
//	finalwidth = frame.cols * sam_scale;
//	finalheight = frame.rows * sam_scale;
//	//////根据行人框放大倍数和最终的roi尺寸确定roi怎么分割
//	crop_roi.width = static_cast<int>(finalwidth / scale);
//	crop_roi.height = static_cast<int>(finalheight / scale);
//	if (tracking_roi.x - crop_roi.width / 2 < 0)
//		crop_roi.x = 0;
//	else
//		crop_roi.x = tracking_roi.x - crop_roi.width / 2;
//	if (tracking_roi.y - crop_roi.height / 2 < 0)
//		crop_roi.y = 0;
//	else
//		crop_roi.y = tracking_roi.y - crop_roi.height / 2;
//	if (crop_roi.x + crop_roi.width > frame.cols - 1)
//	{
//		crop_roi.width = frame.cols - 1 - crop_roi.x;
//		scale = static_cast<float>(finalwidth) / static_cast<float>(crop_roi.width);
//	}
//	if (crop_roi.y + crop_roi.height > frame.rows - 1)
//	{
//		crop_roi.height = frame.rows - 1 - crop_roi.y;
//		scale = static_cast<float>(finalheight) / static_cast<float>(crop_roi.height);
//	}
//	return 0;
//}
//
//cv::Mat Cost::trackingCrop(cv::Mat frame)
//{
//	///////resize
//	cv::Mat temp;
//	resize(frame(crop_roi), temp, Size(finalwidth, finalheight));
//	tracking_roi.x = (tracking_roi.x - crop_roi.x) * scale;
//	tracking_roi.y = (tracking_roi.y - crop_roi.y) * scale;
//	tracking_roi.width *= scale;
//	tracking_roi.height *= scale;
//
//
//	return temp;
//}
//
////////Recover the enlarged roi into normal
//int Cost::trackingRecover()
//{
//	tracking_roi.x = static_cast<int>(tracking_roi.x / scale) + crop_roi.x;
//	tracking_roi.y = static_cast<int>(tracking_roi.y / scale) + crop_roi.y;
//	tracking_roi.width = static_cast<int>(tracking_roi.width / scale);
//	tracking_roi.height = static_cast<int>(tracking_roi.height / scale);
//
//	return 0;
//}
//
//int Cost::tracking_init(cv::Mat frame)
//{
//	cv::Mat img;
//	trackingSetBlock(frame);
//	trackingCrop(frame).copyTo(img);
//	KCF_tracker.init(tracking_roi, img);
//	istracking = 1;
//
//	return 0;
//}
//
//bool Cost::tracking(cv::Mat frame)
//{
//	cv::Mat img;
//	trackingCrop(frame).copyTo(img);
//	tracking_roi = KCF_tracker.update(img);
//	
//	trackingRecover();
//	cv::rectangle(show_opencv, tracking_roi, cv::Scalar(0, 0, 255), 5);
//	resize(show_opencv, show_opencv, Size(800, 600));
//	imshow("ref", show_opencv);
//	cv::waitKey(30);
//
//	return 1;
//}

int Cost::tracking_init(cv::Mat frame)
{
	cv::Mat temp;
	cv::resize(frame, temp, Size(frame.cols * 2, frame.rows * 2));
	tracking_roi.x *= 2;
	tracking_roi.y *= 2;
	tracking_roi.width *= 2;
	tracking_roi.height *= 2;
	KCF_tracker.init(tracking_roi, temp);

	//////recover
	tracking_roi.x /= 2;
	tracking_roi.y /= 2;
	tracking_roi.width /= 2;
	tracking_roi.height /= 2;

	return 0;
}

bool Cost::tracking(cv::Mat frame)
{
	cv::Mat temp;
	cv::resize(frame, temp, Size(frame.cols * 2, frame.rows * 2));
	tracking_roi = KCF_tracker.update(temp);
	tracking_roi.x /= 2;
	tracking_roi.y /= 2;
	tracking_roi.width /= 2;
	tracking_roi.height /= 2;
	cv::rectangle(show_opencv, tracking_roi, cv::Scalar(0, 0, 255), 5);
	resize(show_opencv, show_opencv, Size(800, 600));
	imshow("ref", show_opencv);
	cv::waitKey(30);

	return isfind();
}

int Cost::Thtracking()
{
	while (1)
	{
		cv::Mat local_bayer, ref_bayer;
		cv::Mat watching;
		std::vector<cam::Imagedata> imgdatas(2);
		cameraPtr->captureFrame(imgdatas);
		cv::Mat(camInfos[0].height, camInfos[0].width,
			CV_8U, reinterpret_cast<void*>(imgdatas[0].data)).copyTo(local_bayer);
		cv::Mat(camInfos[1].height, camInfos[1].width,
			CV_8U, reinterpret_cast<void*>(imgdatas[1].data)).copyTo(ref_bayer);
		cv::Mat local, ref;
		//////////////convert/////////////
		//cv::cvtColor(local_bayer, local, CV_BayerRG2BGR);
		cv::cvtColor(ref_bayer, ref, CV_BayerRG2BGR);
		std::vector<cv::Mat> channels(3);
		/*split(local, channels);
		channels[0] = channels[0] * camInfos[0].blueGain;
		channels[1] = channels[1] * camInfos[0].greenGain;
		channels[2] = channels[2] * camInfos[0].redGain;
		merge(channels, local);*/

		split(ref, channels);
		channels[0] = channels[0] * camInfos[1].blueGain;
		channels[1] = channels[1] * camInfos[1].greenGain;
		channels[2] = channels[2] * camInfos[1].redGain;
		merge(channels, ref);

		cv::Mat temp;
		cv::resize(ref, temp, Size(ref.cols * 2, ref.rows * 2));
		tracking_roi = KCF_tracker.update(temp);
		tracking_roi.x /= 2;
		tracking_roi.y /= 2;
		tracking_roi.width /= 2;
		tracking_roi.height /= 2;
		ref.copyTo(show_opencv2);
		cv::rectangle(show_opencv2, tracking_roi, cv::Scalar(0, 0, 255), 5);
		thread_flag = 1;  //可以imshow show_opencv2了
		if (isfind() == 0 || Thread_end == 1)  //
		{
			isfind_time = 0;  //检测计数器清零
			istracking = 0;  //
			Thread_end = 0;
			std::cout << "thread exit!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
			return 0;   //退出线程，准备下一次线程开始
		}
	}
}

void Cost::startTh()
{
	std::cout << "thread into!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
	std::thread Th_tracking(&Cost::Thtracking, this);
	Th_tracking.detach();
}

bool Cost::isfind()
{
	isfind_vec[isfind_time] = cv::Point(tracking_roi.x, tracking_roi.y);
	isfind_time++;
	if (isfind_time == isfind_max)
	{
		if (abs(isfind_vec[isfind_time - 1].x - isfind_vec[0].x) < 3
			&& abs(isfind_vec[isfind_time - 1].y - isfind_vec[0].y) < 3)
		{
			isfind_time = 0;
			return 0;
		}
		isfind_time = 0;
	}
	return 1;
}

bool Cost::iscontain(bbox_t roi)
{
	if (tracking_roi.x > roi.x && tracking_roi.y > roi.y
		&& tracking_roi.x + tracking_roi.width < roi.x + roi.w
		&& tracking_roi.y + tracking_roi.height < roi.y + roi.h)
		return 1;
	return 0;
}

std::vector<bbox_t> Cost::detection(cv::Mat img)
{
	float ratio_width;
	float ratio_height;
	std::vector<bbox_t> people_vec;
	std::vector<bbox_t> final_vec;
	cv::Mat temp;
	img.copyTo(temp);
	ratio_width = static_cast<float>(temp.cols) / 416.0f;
	ratio_height = static_cast<float>(temp.rows) / 416.0f;
	cv::resize(temp, temp, cv::Size(416, 416));

	cv::Mat imgf;
	cv::cvtColor(temp, imgf, cv::COLOR_BGR2RGB);
	imgf.convertTo(imgf, CV_32F, 1.0 / 255.0);

	float *img_h = new float[imgf.rows * imgf.cols * 3];
	size_t count = 0;
	for (size_t k = 0; k < 3; ++k) {
		for (size_t i = 0; i < imgf.rows; ++i) {
			for (size_t j = 0; j < imgf.cols; ++j) {
				img_h[count++] = imgf.at<cv::Vec3f>(i, j).val[k];
			}
		}
	}

	float* img_d;
	cudaMalloc(&img_d, sizeof(float) * 3 * temp.rows * temp.cols);
	cudaMemcpy(img_d, img_h, sizeof(float) * 3 * temp.rows * temp.cols,
		cudaMemcpyHostToDevice);


	detector->detect(img_d, temp.cols, temp.rows, people_vec);
	for (size_t i = 0; i < people_vec.size(); i++) {
		people_vec[i].x *= ratio_width;
		people_vec[i].y *= ratio_height;
		people_vec[i].w *= ratio_width;
		people_vec[i].h *= ratio_height;
		if (people_vec[i].prob > 0.25 && people_vec[i].obj_id == 0)
			final_vec.push_back(people_vec[i]);
	}
	//如果没检测到人，就随机选一块区域显示
	if (final_vec.size() == 0)
	{
		final_vec.resize(1);
		final_vec[0].x = 1000;
		final_vec[0].y = 750;
		final_vec[0].w = 200;
		final_vec[0].h = 100;
	}

	return final_vec;
}

//according to the ref tracking, to choose the people in local
cv::Mat Cost::SetFaceBlock(cv::Mat ref_people,cv::Mat local)
{
	std::vector<bbox_t> vec = detection(local);
	std::cout << "in local, we detect " << vec.size() << " people!!" << std::endl;
	cv::Mat output;
	//检测带人脸的行人，如果没有，则标志位置零
	for (int z = 0; z < vec.size(); z++)
	{
		cv::Rect rect(vec[z].x, vec[z].y, vec[z].w, vec[z].h);
		if (rect.x - 50 > 0 && rect.y - 50 > 0
			&& rect.x + rect.width + 50 < local.cols - 1
			&& rect.y + rect.height + 50 < local.rows - 1)
		{
			rect.x -= 50;
			rect.y -= 50;
			rect.width += 100;
			rect.height += 100;
		}

		rect.height /= 2;
		cv::Mat img;
		local(rect).copyTo(img);
		img.copyTo(output);
		std::vector<bbox_t> face_vec;

		float ratio_width;
		float ratio_height;
		ratio_width = static_cast<float>(img.cols) / 416.0f;
		ratio_height = static_cast<float>(img.rows) / 416.0f;
		cv::resize(img, img, cv::Size(416, 416));

		cv::Mat imgf;
		cv::cvtColor(img, imgf, cv::COLOR_BGR2RGB);
		imgf.convertTo(imgf, CV_32F, 1.0 / 255.0);

		float *img_h = new float[imgf.rows * imgf.cols * 3];
		size_t count = 0;
		for (size_t k = 0; k < 3; ++k) {
			for (size_t i = 0; i < imgf.rows; ++i) {
				for (size_t j = 0; j < imgf.cols; ++j) {
		img_h[count++] = imgf.at<cv::Vec3f>(i, j).val[k];
				}
			}
		}

		float* img_d;
		cudaMalloc(&img_d, sizeof(float) * 3 * img.rows * img.cols);
		cudaMemcpy(img_d, img_h, sizeof(float) * 3 * img.rows * img.cols,
		cudaMemcpyHostToDevice);


		detector_face->detect(img_d, img.cols, img.rows, face_vec);
		for (size_t i = 0; i < face_vec.size(); i++) 
		{
			face_vec[i].x *= ratio_width;
			face_vec[i].y *= ratio_height;
			face_vec[i].w *= ratio_width;
			face_vec[i].h *= ratio_height;
			if (face_vec[i].prob > 0.1 && face_vec[i].obj_id == 0)
			{
				cv::Rect face(face_vec[i].x, face_vec[i].y, face_vec[i].w, face_vec[i].h);
				output(face).copyTo(current_show[1]);
				find_face = 1;
				return output;
			}
		}
	}

	/*double max = 0.0;
	int index = 0;
	for (int i = 0; i < vec.size(); i++)
	{
		cv::Rect rect(vec[i].x, vec[i].y, vec[i].w, vec[i].h);
		double value = people_match(ref_people, local(rect));
		if (value > max)
		{
			max = value;
			index = i;
		}
	}
*/
	return output;
}

//////detect the face according to the detected people
int Cost::face_detection(cv::Mat local)
{
	cv::Mat Vec_people = SetFaceBlock(ref_people, local); //其实就是找到最有可能的行人框
	if (find_face == 0)
		return -1;
	else
	{
		Vec_people.copyTo(current_show[0]);
		find_face = 0;
		return 0;
	}
}

int Cost::init_face_detection()
{
	std::string cfgfile = "E:/data/YOLO/yolo-face.cfg";
	std::string weightfile = "E:/data/YOLO/yolo-face.weights";
	detector_face->init(cfgfile, weightfile, 0);
	return 0;
}

double Cost::people_match(cv::Mat img1, cv::Mat img2)
{
	calib::FeatureMatch match;
	cv::Mat temp1, temp2;
	img1.copyTo(temp1);
	img2.copyTo(temp2);
	resize(temp1, temp1, cv::Size(temp2.cols, temp2.rows));
	double confidence = match.match_people(temp1, temp2);
	return confidence;
}
