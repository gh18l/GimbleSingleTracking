// include std
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include<fstream>
#include <queue>
#include <thread>
#include <memory>
#include <time.h>
#include "Serial.h"  
#include <string.h> 
#include<tchar.h>
// opencv
#include <opencv2/opencv.hpp>
#include "Display.h"
//
#include "opencv2/core.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
using namespace std;
using namespace cv;

// cuda
#ifdef _WIN32
#include <windows.h>
#endif
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "GenCameraDriver.h"
#include "TinyStitcher.h"
#include "Cost.h"
#include "FeatureMatch.h"
#include "CameraParamEstimator.h"
#include "Compositor.h"

//imshow
//#define _SHOW

//#define DEBUG_
//#define SAVEAVI

#define coe_gain 2.5


CSerial serial;
TinyStitcher stitcher;
Cost cost;
Display display;

calib::FeatureMatch match;
calib::CameraParamEstimator estimator;
calib::BundleAdjustment bundleAdjust;
calib::Compositor compositor;


std::vector<cam::GenCamInfo> camInfos;
std::shared_ptr<cam::GenCamera> cameraPtr;
int num=0;
cv::VideoWriter writer1, writer2;
bool start_flag = 0;
bool startTracking_flag = 0;

int delay_gimble()
{
	SysUtil::sleep(1500);
	return 0;
}
int collectdelay_gimble()
{
	SysUtil::sleep(2500);
	return 0;
}

void shoot(cv::Mat &ref, cv::Mat &local)
{
	char name[200], temp[50];  //
	cv::Mat local_bayer, ref_bayer;
	cv::Mat watching;
	std::vector<cam::Imagedata> imgdatas(2);
	cameraPtr->captureFrame(imgdatas);
	cv::Mat(camInfos[0].height, camInfos[0].width,
		CV_8U, reinterpret_cast<void*>(imgdatas[0].data)).copyTo(local_bayer);
	cv::Mat(camInfos[1].height, camInfos[1].width,
		CV_8U, reinterpret_cast<void*>(imgdatas[1].data)).copyTo(ref_bayer);

	//////////////convert/////////////
	cv::cvtColor(local_bayer, local, CV_BayerRG2BGR);
	cv::cvtColor(ref_bayer, ref, CV_BayerRG2BGR);
	std::vector<cv::Mat> channels(3);
	split(local, channels);
	channels[0] = channels[0] * camInfos[0].blueGain;
	channels[1] = channels[1] * camInfos[0].greenGain;
	channels[2] = channels[2] * camInfos[0].redGain;
	merge(channels, local);

	split(ref, channels);
	channels[0] = channels[0] * camInfos[1].blueGain;
	channels[1] = channels[1] * camInfos[1].greenGain;
	channels[2] = channels[2] * camInfos[1].redGain;
	merge(channels, ref);
#ifdef SAVEAVI
	writer1 << ref;
	writer2 << local;
#endif
#ifdef _SHOW
	cv::Mat show1, show2;
	cv::resize(local, show1, cv::Size(800, 600));
	cv::resize(ref, show2, cv::Size(800, 600));
	cv::imshow("local", show1);
	cv::imshow("ref", show2);
	cv::waitKey(30);
#endif
}

void gimbal_init(float delta_Yaw, float delta_Pitch)
{
	stitcher.current_pulse.x = delta_Yaw + YM;
	stitcher.current_pulse.y = delta_Pitch + PM;
	serial.Serial_Send_Yaw(YM + delta_Yaw);     //这里用类的静态成员更好
	serial.Serial_Send_Pitch(PM + delta_Pitch);
	collectdelay_gimble();
}

void init(cv::Point init)    //别忘了关相机
{
	// init camera
	cameraPtr = cam::createCamera(cam::CameraModel::XIMEA_xiC);
	cameraPtr->init();
	// set camera setting
	cameraPtr->startCapture();
	cameraPtr->setFPS(-1, 20);
	cameraPtr->setAutoExposure(-1, cam::Status::on);
	cameraPtr->setAutoExposureLevel(-1, 40);
	cameraPtr->setAutoWhiteBalance(-1);
	cameraPtr->makeSetEffective();
	// set capturing setting
	cameraPtr->setCamBufferType(cam::GenCamBufferType::Raw);
	cameraPtr->setCaptureMode(cam::GenCamCaptureMode::Continous, 40);
	cameraPtr->setCapturePurpose(cam::GenCamCapturePurpose::Streaming);
	//cameraPtr->setVerbose(true);
	// get camera info
	cameraPtr->getCamInfos(camInfos);
	cameraPtr->startCaptureThreads();
	cameraPtr->makeSetEffective();

	//init serial
	serial.OpenSerialPort(_T("COM6:"), 9600, 8, 1);

	//init gimble
	stitcher.init_stitcher();
	cv::Mat ref, local;
	gimbal_init(init.x, init.y);    //转到0,0脉冲处
	shoot(ref, local);
	stitcher.find_position(ref, local, stitcher.current_point);    //update now point
	std::cout << stitcher.current_point << std::endl;
	//init picture parameters


}
void camera_close()
{
	cameraPtr->stopCaptureThreads();
	cameraPtr->release();
}

void collection(std::vector<cv::Mat>& imgs)
{
	std::string datapath = "E:/data/bb";
	cv::Mat ref, local;
	stitcher.current_pulse.x = YM+180;
	stitcher.current_pulse.y = PM+120;
	serial.Serial_Send_Yaw(stitcher.current_pulse.x);
	serial.Serial_Send_Pitch(stitcher.current_pulse.y);
	collectdelay_gimble();

	for (int i = 0; i < stitcher.Row; i++)
	{
		if (i % 2 == 0)
		{
			for (int j = 0; j < stitcher.Col; j++)
			{
				stitcher.current_pulse.x = YM + 180 - stitcher.dX * j;
				stitcher.current_pulse.y = PM + 120 - stitcher.dY * i;
				serial.Serial_Send_Yaw(stitcher.current_pulse.x);
				serial.Serial_Send_Pitch(stitcher.current_pulse.y);
				collectdelay_gimble();
				for (int k = 0; k < 1; k++)
				{
					shoot(ref, local);
					local.copyTo(imgs[j*stitcher.Row + i]);
					cv::imwrite(cv::format("%s/local_%d_%d.png",
						datapath.c_str(), int(stitcher.dX) * j, int(stitcher.dY) * i), local);
				}
			}
		}
		
		if (i % 2 == 1)
		{
			for (int j = stitcher.Col-1; j >= 0; j--)
			{
				stitcher.current_pulse.x = YM + 180 - stitcher.dX * j;
				stitcher.current_pulse.y = PM + 120 - stitcher.dY * i;
				serial.Serial_Send_Yaw(stitcher.current_pulse.x);
				serial.Serial_Send_Pitch(stitcher.current_pulse.y);
				collectdelay_gimble();
				for (int k = 0; k < 1; k++)
				{
					shoot(ref, local);
					local.copyTo(imgs[j*stitcher.Row + i]);
					cv::imwrite(cv::format("%s/local_%d_%d.png",
						datapath.c_str(), int(stitcher.dX) * j, int(stitcher.dY) * i), local);
				}
			}
		}
	}
}


bool isOverlap(const cv::Rect &rc1, const cv::Rect &rc2)
{
	if (rc1.x + rc1.width  > rc2.x &&
		rc2.x + rc2.width  > rc1.x &&
		rc1.y + rc1.height > rc2.y &&
		rc2.y + rc2.height > rc1.y
		)
		return true;
	else
		return false;
}
int findoverlap(cv::Point corner_current, cv::Size size_current, vector<Point>& corners, vector<Size>& sizes, std::vector<int>& index)
{
	cv::Rect Rect_current(corner_current, size_current);
	for (int i = 0; i < stitcher.Row*stitcher.Col; i++)
	{
		cv::Rect temp(corners[i], sizes[i]);
		if (isOverlap(Rect_current, temp))
		{
			index.push_back(i);   //存放着相连图像的序号，用过后别忘了清空
		}
	}
	return 0;
}
int save_para(vector<calib::CameraParams>& cameras, vector<Point>& corners, vector<Size>& sizes)
{
	ofstream para;
	Mat K;
	para.open("E:/code/project/gimble3.23/para.txt", ios::out);
	if (!para)
		cout << "No have txt" << endl;
	for (int i = 0; i < cameras.size(); i++)
	{
		para << cameras[i].focal << " " << cameras[i].aspect << " "
			<< cameras[i].ppx << " " << cameras[i].ppy << " ";
		//可以考虑看下Mat_模板类，K()const函数被定义在camera.cpp里
		for (int j = 0; j < 3; j++)
		{
			for (int k = 0; k < 3; k++)
			{
				para << cameras[i].R.at<double>(j, k) << " ";
			}
		}
		para << corners[i].x << " " << corners[i].y << " " << sizes[i].width << " " << sizes[i].height << " ";
		/*for (int j = 0; j < 3; j++)
		{
		para << cameras[i].t.at<float>(j,0) << " ";
		}*/
		para << endl;
	}
	para.close();

	return 0;
}

int read_para(vector<calib::CameraParams> &cameras, vector<Point> &corners, vector<Size>&sizes)
{
	ifstream para;
	Mat K;
	para.open("E:/code/project/gimble3.23/para.txt");
	if (!para.is_open())
	{
		cout << "can not open txt" << endl;
		return -1;
	}
	string str;

	for (int i = 0; i < stitcher.Row*stitcher.Col; i++)   //这里没有自动计算图片个数！！！！！！
	{
		getline(para, str, ' ');
		cameras[i].focal = stof(str);
		getline(para, str, ' ');
		cameras[i].aspect = stof(str);
		getline(para, str, ' ');
		cameras[i].ppx = stof(str);
		getline(para, str, ' ');
		cameras[i].ppy = stof(str);
		cameras[i].R.create(3, 3, CV_64FC1);
		for (int j = 0; j < 3; j++)
		{
			for (int k = 0; k < 3; k++)
			{
				getline(para, str, ' ');
				cameras[i].R.at<double>(j, k) = stof(str);
			}
		}
		getline(para, str, ' ');
		corners[i].x = stoi(str);
		getline(para, str, ' ');
		corners[i].y = stoi(str);
		getline(para, str, ' ');
		sizes[i].width = stoi(str);
		getline(para, str, ' ');
		sizes[i].height = stoi(str);
		/*for (int j = 0; j < 3; j++)
		{
		getline(para, str, ' ');
		cameras[i].t.at<float>(j, 0) = stof(str);
		}*/
		getline(para, str);
	}
	para.close();
	return 0;
}

int GetCurrentPara(vector<calib::CameraParams>& cameras, cv::Point2f current_point, calib::CameraParams &current_para)
{
	///build a para matrix
	cv::Mat focals(stitcher.Row, stitcher.Col, CV_64FC1), ppxs(stitcher.Row, stitcher.Col, CV_64FC1), ppys(stitcher.Row, stitcher.Col, CV_64FC1);
	int index = 0;
	//////R and T
	vector<cv::Mat>Rs(9);   //每层对应一个r元素, 每层元素的个数对应图片个数
	for (int i = 0; i < 9; i++)
	{
		Rs[i].create(stitcher.Row, stitcher.Col, CV_64FC1);
	}
	//vector<cv::Mat>Ts(3);
	for (int i = 0; i < stitcher.Col; i++)      //从上往下读数据
	{
		for (int j = 0; j < stitcher.Row; j++)
		{
			focals.at<double>(j, i) = cameras[index].focal;      //第j行第i列
			ppxs.at<double>(j, i) = cameras[index].ppx;
			ppys.at<double>(j, i) = cameras[index].ppy;
			////每运行一组kl循环代表矩阵一个位置被填上 深度为9
			for (int k = 0; k < 3; k++)
			{
				for (int l = 0; l < 3; l++)
				{
					Rs[k * 3 + l].at<double>(j, i) = cameras[index].R.at<double>(k, l); //
				}
			}
			index++;
		}
	}

	///////////////上面的最后还是要加到readpara函数里的
	vector<double>value;
	vector<float>mapx(1, (current_point.x - stitcher.X_MIN)*1.0 / stitcher.dX);    //默认坐标从00开始
	vector<float>mapy(1, (current_point.y - stitcher.Y_MIN)*1.0 / stitcher.dY);    //默认坐标从00开始
	remap(focals, value, mapx, mapy, INTER_LINEAR);    //可能需要clear一下，得实验返回值是push还是覆盖
	current_para.focal = value[0];
	value.clear();

	remap(ppxs, value, mapx, mapy, INTER_LINEAR);
	current_para.ppx = value[0];
	value.clear();

	remap(ppys, value, mapx, mapy, INTER_LINEAR);
	current_para.ppy = value[0];
	value.clear();
	current_para.R.create(3, 3, CV_64FC1);
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			remap(Rs[i * 3 + j], value, mapx, mapy, INTER_LINEAR);
			current_para.R.at<double>(i, j) = value[0];
			value.clear();
		}
	}
	return 0;
}
int panoram(cv::Mat& result,std::vector<cv::Mat>& imgs)
{
	collection(imgs);
	int img_num = imgs.size();
	std::cout << img_num << std::endl;
	cv::Mat connection;
	connection = cv::Mat::zeros(img_num, img_num, CV_8U);
	for (size_t i = 0; i < imgs.size(); i++) {
		for (size_t j = 0; j < imgs.size(); j++) {
			if (i == j)
				continue;
			int row1 = i / stitcher.Row;
			int col1 = i % stitcher.Row;
			int row2 = j / stitcher.Row;
			int col2 = j % stitcher.Row;
			if (abs(row1 - row2) <= 1 && abs(col1 - col2) <= 1) {
				connection.at<uchar>(i, j) = 1;
				connection.at<uchar>(j, i) = 1;
			}
		}
	}

	match.init(imgs, connection);
	match.match();
	//match.debug();

	calib::CameraParamEstimator estimator;
	estimator.init(imgs, connection, match.getImageFeatures(), match.getMatchesInfo());
	estimator.estimate();
	
	calib::Compositor compositor;
	compositor.init(imgs, estimator.getCameraParams());
	compositor.composite();
	save_para(compositor.cameras, compositor.corners, compositor.sizes);

	return 0;
}
int warp(std::vector<cv::Mat>& imgs, vector<calib::CameraParams> &cameras, 
			vector<Point> &corners, vector<Size>&sizes, cv::Mat& src, cv::Point2f current_pulse, 
				calib::CameraParams& current_para, std::vector<calib::Imagefeature>& features)
{
	cv::Ptr<cv::detail::SphericalWarper> w = cv::makePtr<cv::detail::SphericalWarper>(false);
	std::shared_ptr<cv::detail::Blender> blender_ = std::make_shared<cv::detail::MultiBandBlender>(false);
	clock_t start, finish;
	current_pulse.x = YM + 180 - current_pulse.x;
	current_pulse.y = PM + 120 - current_pulse.y;
	GetCurrentPara(cameras, current_pulse, current_para);
	/////////求出corner_current和size_current///////////////
	cv::Mat src_warped, mask, mask_warped;
	cv::Point corner_current;
	cv::Size size_current;
	w->setScale(16000);
	// calculate warping filed
	cv::Mat K, R;
	current_para.K().convertTo(K, CV_32F);
	current_para.R.convertTo(R, CV_32F);
	cv::Mat initmask(src.rows, src.cols, CV_8U);
	initmask.setTo(cv::Scalar::all(255));
	corner_current = w->warp(src, K, R, cv::INTER_LINEAR, cv::BORDER_CONSTANT, src_warped);
	w->warp(initmask, K, R, cv::INTER_NEAREST, cv::BORDER_CONSTANT, mask_warped);
	size_current = mask_warped.size();

	//calib::Compositor compositor;
	std::vector<int> index;
	findoverlap(corner_current, size_current, corners, sizes, index);   //求出compositor.index

	calib::FeatureMatch match;
	calib::Imagefeature current_feature;
	std::vector<calib::Matchesinfo> current_matchesInfo(index.size());
	for (int i = 0; i < index.size(); i++)
	{
		match.current_feature_thread_(src, features[index[i]], current_feature, current_matchesInfo[i], i);    //建立起拼接图像和周围图像的特征匹配
	}
	//calib::BundleAdjustment bundleAdjust;
	calib::BundleAdjustment bundleAdjust;
	bundleAdjust.refine_BA(index, current_feature, features, current_matchesInfo, cameras, current_para);    //得到当前current_camera的准确值
#ifdef DEBUG_
	for (int i = 0; i < index.size(); i++)
	{
		/*if (current_matchesInfo[i].confidence < 1.5)
		continue;*/
		if (current_matchesInfo[i].confidence < 2.9)
			continue;
		cv::Mat result;
		// make result image
		int width = src.cols * 2;
		int height = src.rows;
		result.create(height, width, CV_8UC3);
		cv::Rect rect(0, 0, src.cols, height);
		src.copyTo(result(rect));
		rect.x += src.cols;
		imgs[index[i]].copyTo(result(rect));
		// draw matching points
		cv::RNG rng(12345);
		int r = 3;
		for (size_t kind = 0; kind < current_matchesInfo[i].matches.size(); kind++) {
			if (current_matchesInfo[i].inliers_mask[kind]) {
				cv::Scalar color = cv::Scalar(rng.uniform(0, 255),
					rng.uniform(0, 255), rng.uniform(0, 255));
				const cv::DMatch& m = current_matchesInfo[i].matches[kind];
				cv::Point2f p1 = current_feature.keypt[m.queryIdx].pt;
				cv::Point2f p2 = features[i].keypt[m.trainIdx].pt;
				p2.x += src.cols;
				cv::circle(result, p1, r, color, -1, cv::LINE_8, 0);
				cv::circle(result, p2, r, color, -1, cv::LINE_8, 0);
				//cv::line(result, p1, p2, color, 5, cv::LINE_8, 0);
			}
		}
		cv::imwrite(cv::format("E:/code/project/gimble3.23/features/matching_points_%02d_%02d.jpg", -1, index[i]),
			result);
	}
#endif

	return 0;
}

int get_position(cv::Mat refImg,cv::Mat localImg,cv::Rect& refRect)
{
	cv::Mat refEdge = stitcher.SEDDetector(refImg, 0.25);
	cv::Mat localEdge = stitcher.SEDDetector(localImg, 0.25);
	// resize localview image
	cv::Mat templ, templEdge;
	stitcher.sizeBlock = cv::Size(localImg.cols * 0.118, localImg.rows * 0.118);
	cv::resize(localImg, templ, stitcher.sizeBlock);
	cv::resize(localEdge, templEdge, stitcher.sizeBlock);

	cv::Mat result, resultEdge;
	cv::matchTemplate(refImg, templ, result, cv::TM_CCOEFF_NORMED);
	cv::matchTemplate(refEdge, templEdge, resultEdge, cv::TM_CCOEFF_NORMED);
	result = result.mul(resultEdge);

	cv::Point maxLoc;
	cv::minMaxLoc(result, NULL, NULL, NULL, &maxLoc);
	refRect = cv::Rect(maxLoc.x, maxLoc.y, stitcher.sizeBlock.width, stitcher.sizeBlock.height);

	return 0;
}

int main(int argc, char* argv[]) {
	clock_t start, finish;

	cv::Mat result;
	std::vector<cv::Mat> imgs;
	vector<calib::CameraParams> cameras(stitcher.Row*stitcher.Col);
	vector<Point> corners(stitcher.Row*stitcher.Col);
	vector<Size> sizes(stitcher.Row*stitcher.Col);
	cv::Mat ref, local;
	init(cv::Point(0, 0));
	if (argc > 1)//有参数，有可能不需要扫描了
	{
		std::string panoname = std::string(argv[1]);
		result = cv::imread(panoname);
		std::string datapath = "E:/data/bb";
		for (int i = 0; i <= 280; i = i + 40) {
		for (int j = 0; j <= 240; j = j + 30) {
			imgs.push_back(cv::imread(cv::format("%s/local_%d_%d.png", 
				datapath.c_str(), i, j)));
			}
		}
	}
	else
	{
		imgs.resize(stitcher.Row*stitcher.Col);
		panoram(result, imgs);
	}
	//////////read parameters///////////
	read_para(cameras, corners, sizes);
	std::vector<calib::Imagefeature> features;
	match.read_features(features);
	//global-result warping
	cv::Mat result_temp;

	cv::Point dst_point,src_point;
	serial.Serial_Send_Yaw(YM);
	serial.Serial_Send_Pitch(PM);
	stitcher.current_pulse.x = YM;
	stitcher.current_pulse.y = PM;
	collectdelay_gimble();
	shoot(ref, local);
	stitcher.find_position(ref, local, stitcher.current_point);
	src_point = stitcher.current_point;

	corners.resize(cameras.size() + 1);
	sizes.resize(cameras.size() + 1);
	for (int i = 0; i < 100; i++)
	{
		cost.video_preflow(ref);
		shoot(ref, local);
		std::cout << i << std::endl;
	}
	calib::CameraParams current_para;
	calib::Compositor compositor;
	display.display_init(result);
	cost.init_people_detection(ref);
	cost.init_face_detection();
	cv::Mat ref_temp;
	int index = 0;

	while (1)
	{
		result.copyTo(result_temp);
		shoot(ref, local);
		ref.copyTo(ref_temp);
		stitcher.colorCorrectRGB(local, result_temp);
		if (index > 50 && cost.video_updatedetection(ref_temp, src_point, dst_point) == 0 
			&& stitcher.detect_move(dst_point) == 0)
		{
			if (cost.istracking == 0) //需要更新tracking_roi，并把ref人取出来
			{
				//更新tracking_roi，初始化tracking
				cost.tracking_roi.x = cost.current_vec.x;
				cost.tracking_roi.y = cost.current_vec.y;
				cost.tracking_roi.width = cost.current_vec.w;
				cost.tracking_roi.height = cost.current_vec.h;
				cost.tracking_init(ref_temp);
				cost.istracking = 1;
				//把人取出来
				ref_temp(cost.tracking_roi).copyTo(cost.ref_people);
				cost.startTh();

				//转
				stitcher.move(dst_point);   //更新current_pulse和current_point
				serial.Serial_Send_Yaw(stitcher.current_pulse.x);
				serial.Serial_Send_Pitch(stitcher.current_pulse.y);
				delay_gimble();
				//拍一张
				shoot(ref, local);
				//找
				stitcher.gimble_find_position(ref, local,
					stitcher.current_point, 2, stitcher.current_point);
				src_point = stitcher.current_point;
				ref.copyTo(ref_temp);
				cv::Mat local_temp;
				//resize(local, local_temp,
					//Size(800,600));
				//cv::imshow("local", local_temp);
				stitcher.colorCorrectRGB(local, result_temp);
				//在local里检测人，对比并将local人抠出
				//抠出的高清人放在current_show里，目前是行人和人脸
				if (cost.face_detection(local, ref_temp,stitcher.current_point) == -1)   //没检测到人脸
				{
					std::cout << "not find face!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
					//结束线程，结束跟踪
					cost.Thread_end = 1;
					continue;
				}
				 
				//算相机参数
				warp(imgs, cameras, corners, sizes, local, stitcher.current_pulse,
					current_para, features);   
				//显示更新的current_show
				//cv::Mat people_temp, face_temp;
				//resize(cost.current_show[0], people_temp, 
					//Size(cost.current_show[0].cols, cost.current_show[0].rows));
				//resize(cost.current_show[1], face_temp,
					//Size(cost.current_show[1].cols * 4, cost.current_show[1].rows * 4));
				//cv::imshow("current people", people_temp);
				//cv::imshow("current face", face_temp);
				//cv::waitKey(30);
				startTracking_flag = 1;
			}

			else //不需要更新tracking_roi
			{
				//转
				stitcher.move(dst_point);   //更新current_pulse和current_point
				serial.Serial_Send_Yaw(stitcher.current_pulse.x);
				serial.Serial_Send_Pitch(stitcher.current_pulse.y);
				delay_gimble();
				//拍一张
				shoot(ref, local);
				ref.copyTo(ref_temp);
				stitcher.colorCorrectRGB(local, result_temp);
				//算相机参数
				warp(imgs, cameras, corners, sizes, local, stitcher.current_pulse,
					current_para, features);
				//找
				stitcher.gimble_find_position(ref, local,
					stitcher.current_point, 2, stitcher.current_point);
				src_point = stitcher.current_point;
			}

			index = 0;
		}
		if (start_flag == 0)
		{
			warp(imgs, cameras, corners, sizes, local, stitcher.current_pulse,
				current_para, features);
			start_flag = 1;
		}
		//用参数拼成result_temp
		compositor.single_composite(current_para, local, result_temp, corners, sizes);
		//显示
		resize(result_temp, result_temp, cv::Size((result_temp.cols / 100) * 100, (result_temp.rows / 100) * 100));
		if (startTracking_flag == 0)
		{
			cv::Mat black = cv::Mat::zeros(600, 600, CV_8UC3);
			display.display(result_temp, black);
		}
		else
		{
			cv::Mat people_temp;
			cv::resize(cost.current_show[0], people_temp, cv::Size(600, 600));
			display.display(result_temp, people_temp);
		}
		
		if (cost.thread_flag == 1)   //进入过一次tracking线程
		{
			cv::Mat tracking_temp;
			resize(cost.show_opencv2, tracking_temp, Size(800, 600));
			imshow("tracking", tracking_temp);
			cv::waitKey(30);
		}
		//各种参数的更新
		index++;
	}
	camera_close();   //////////////////////
	return 0;
}
