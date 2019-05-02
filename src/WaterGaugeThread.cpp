#include "WaterGaugeThread.h"

#include "Analysis.h"
#include "DefineColor.h"
#include <glog/logging.h>


#define JUMP_COUNT 8 
#define LINE_COUNT 10 
#define LEFT_RIGHT_COUNT 5
#define SCALER_COUNT 1

WaterGaugeThread::WaterGaugeThread(Analysis* analysis_manager) 
    : manager(analysis_manager) {

}
void WaterGaugeThread::Run() {
    std::lock_guard<std::mutex> lk(_mutex);
    LOG(INFO) << "WaterGaugeThread start";
    _is_run = true;
    uint32_t interval = config.detect_interval() * 1000;
    while (_is_run) {
        cv::Mat origin_img = manager->GetOriginImg();
        cv::Mat segment_img = manager->GetSegmentImg();
//cv::Mat origin_img;
//if(!temp.empty())
//{
//        resize(temp, origin_img, cv::Size(temp.cols/3,temp.rows/3));
 //       cv::imshow("origin_img", origin_img);
//	cv::waitKey(0);
   //     cv::imshow("segment_img", segment_img);
//	imshow("canny1", segment_img);
//}
//else
//{
//origin_img=temp.clone();
//}
	//cv::waitKey(0);
	std::cout<<"origin_img"<<std::endl;
        cv::imwrite("./origin_img.png",origin_img);
        cv::imwrite("./segment_img.png",segment_img);
        if (origin_img.empty() || segment_img.empty()) {
            usleep(interval);
            continue;
        }

#ifdef _DEBUG
        if (config.has_display() && config.display()) {
            std::string window_name = std::to_string(manager->Config().video_id());
            std::string origin_name = window_name + std::string("_WaterGauge_origin");
            std::string segment_name = window_name + std::string("_WaterGauge_segment");
            cv::imshow(origin_name, origin_img);
            cv::imshow(segment_name, segment_img);
            cv::waitKey(1);
        }
#endif  
        int32_t result = 0;
        ErrorType state = CWaterLevelMeasure(origin_img, segment_img, result);
	std::cout<<"state"<<state<<std::endl;
        std::cout<<"result"<<result<<std::endl;
	    if (state == Normal)
    	{
		    float numberOfRealScaler = 1 - (result / 0.6)*0.01;
            SetAlarm(numberOfRealScaler);
    	}
        
        usleep(interval);
    }
    LOG(INFO) << "WaterGaugeThread end";
    _is_run = false;
    CallStop();
}

AnalysisAlarm WaterGaugeThread::GetAlarm() {
    std::lock_guard<std::mutex> lk(alarm_mutex);
    AnalysisAlarm res = alarm;
    alarm.set_is_active(false);
    return res;
}

void WaterGaugeThread::SetAlarm(float num) {
    std::lock_guard<std::mutex> lk(alarm_mutex);
    alarm.set_is_active(true);
    alarm.set_water_gauge_num(num);
}

ErrorType WaterGaugeThread::CWaterLevelMeasure(cv::Mat inputMatSrc, cv::Mat segmentMat,int &sraftNumber)
{
	draft._lastError = Normal;
	if (!inputMatSrc.data || !(inputMatSrc.channels() == 3 || inputMatSrc.channels() == 4))
	{
		draft._lastError = InputFileError;
	}
	if (!segmentMat.data || !(segmentMat.channels() == 3 || segmentMat.channels() == 4))
	{
		draft._lastError = InputFileError;
	}
	inputMatSrc.copyTo(draft._inputMat);
	segmentMat.copyTo(draft._segmentMat);
	draft.DraftColor = RedDraft;
	//std::cout<<"origin_img"<<std::endl;
	draft._lastError=getWaterLevelScaler(sraftNumber);
	//std:::cout<<"CWaterLevelMeasure"<<draft._lastError<<std::endl;
	return draft._lastError;
}
ErrorType WaterGaugeThread::WaterDraftMeasure(cv::Mat DivisionInput)
{
	cv::Mat segImg;
	DivisionInput.copyTo(segImg);
	inRange(segImg, cv::Scalar(0, 0, 127), cv::Scalar(0, 0, 127), segImg);//寻找颜色
	//inRange(segImg, cv::Scalar(127, 0, 0), cv::Scalar(127, 0, 0), segImg);//寻找颜色
#ifdef VIEWIMAGE
	imshow("inRange", segImg);
#endif
	threshold(segImg, segImg, 100, 255, cv::THRESH_BINARY_INV);//自适应二值化
#ifdef VIEWIMAGE
	imshow("二值化", segImg);
#endif
	Canny(segImg, segImg, 100, 250);
	imshow("canny1", segImg);
#ifdef VIEWIMAGE
#endif

	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	std::vector<cv::Point> contourRect;
	cv::Rect tempRect;
	int maxAreaIndex1 = 0;
	double maxArea1 = 0;
	findContours(segImg, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE, cv::Point());
	std::vector<cv::RotatedRect> box1(contours.size()); //定义最小外接矩形集合
	cvtColor(segImg, segImg, CV_GRAY2RGB);
	if (contours.size() != 0)
	{
		for (int i = 0; i < contours.size(); i++)
		{
			//同种颜色可能会含有多个轮
			box1[i] = cv::minAreaRect(cv::Mat(contours[i]));  //计算每个轮廓最小外接矩形
			double tempArea = box1[i].size.area();
			if (tempArea>maxArea1)
			{
				maxArea1 = tempArea;
				maxAreaIndex1 = i;
			}

#ifdef VIEWIMAGE
		imshow("外接矩形", segImg);
#endif
		}
		tempRect = box1[maxAreaIndex1].boundingRect();
		//if (tempRect.y + tempRect.height >= segImg.rows)
			tempRect.height = segImg.rows - tempRect.y - 1;
		rectangle(segImg, tempRect, cv::Scalar(0, 0, 255));

		draft._tempRect.x = (float)tempRect.x/draft._segmentMat.cols*draft._inputMat.cols-50;
		draft._tempRect.y = (float)tempRect.y / draft._segmentMat.rows*draft._inputMat.rows-0;
		draft._tempRect.width = (float)tempRect.width / draft._segmentMat.cols*draft._inputMat.cols+50;
		draft._tempRect.height = (float)tempRect.height / draft._segmentMat.rows*draft._inputMat.rows;
		draft._inputMat = draft._inputMat(draft._tempRect);
		//cvtColor(draft._inputMat, draft._inputMat, CV_RGB2GRAY);
#ifdef VIEWIMAGE
		imshow("_inputMat", draft._inputMat);
#endif
		//threshold(draft._inputMat, draft._inputMat, 100, 255, THRESH_BINARY_INV);
		//Canny(draft._inputMat, draft._inputMat, 100, 250);
#ifdef VIEWIMAGE
		imshow("canny", draft._inputMat);
#endif
	}
	else
	{
		return   InputFileError;
	}
	return Normal;

}
ErrorType WaterGaugeThread::preDealWithImage(DraftType type)
{

	draft._grayMat.create(draft._inputMat.rows, draft._inputMat.cols, CV_8UC1);

	cv::Mat  imgHsi;
	imgHsi.create(draft._inputMat.rows, draft._inputMat.cols, CV_8UC3);
	cvtColor(draft._inputMat, imgHsi, CV_BGR2HSV);
	std::vector <cv::Mat> vecHsi;
	split(imgHsi, vecHsi);//分离H S I 

						  //按照HSV二值化 分为红色标尺和蓝色标尺
	if (type == RedDraft)
	{
		for (int y = 0; y < vecHsi[0].rows; y++)
		{
			for (int x = 0; x < vecHsi[0].cols; x++)
			{
				if (vecHsi[1].at<uchar>(y, x)>43 && vecHsi[2].at<uchar>(y, x)>46 && (vecHsi[0].at<uchar>(y, x) < 6 || vecHsi[0].at<uchar>(y, x) >156))
				{
					draft._grayMat.at<uchar>(y, x) = 255;
				}
				else
				{
					draft._grayMat.at<uchar>(y, x) = 0;
				}

			}
		}
		std::cout<<"origin_img111"<<std::endl;
		cv::imshow("_grayMat", draft._grayMat);
		cv::waitKey(10);
	}
	else
	{

	}
	return Normal;
}

ErrorType WaterGaugeThread::getWaterLevelScaler(int &sraftNumber)
{
#if  1
	if (WaterDraftMeasure(draft._segmentMat) == InputFileError)
	{
		return InputFileError;
	}
	preDealWithImage(draft.DraftColor);
	draftLocation();
	draft._grayDraftMat = draft._grayCorrectMat(draft._targetRect);
	//cv::imshow("draft._grayDraftMat", draft._grayDraftMat);

#else
	draftLocation();
	_grayDraftMat = _grayCorrectMat(_targetRect);
#endif
	sraftNumber = getScaler(draft._grayDraftMat, draft._targetRect);
	return Normal;
}


int WaterGaugeThread::findUpDownBounding(cv::Mat& inImg, cv::Rect& outRect)
{
	unsigned int width = inImg.cols;
	unsigned int hight = inImg.rows;
	unsigned int lineJumpPoint = 0;
	outRect.x = 0;
	outRect.width = width;
	//从左向右扫描，获取白色（目标）点数大于LINE_COUNT的点，即作为起点（噪声大的图像存在问题）
	for (unsigned int i = 5; i <hight; ++i)
	{
		lineJumpPoint = 0;
		for (unsigned int j = 0; j< width; ++j)
		{
			if (inImg.at<uchar>(i, j) >200)
			{
				lineJumpPoint++;

			}
		}

		if (lineJumpPoint>LINE_COUNT)
		{
			outRect.y = i - 5;//留一定余量
			break;
		}
	}
	//从右向左扫描，获取白色（目标）点数大于LINE_COUNT的点，即作为终点（噪声大的图像存在问题）
	for (unsigned int i = hight - 5; i > 0; i--)
	{
		lineJumpPoint = 0;
		for (unsigned int j = 0; j< width; j++)
		{
			if (inImg.at<uchar>(i, j) >200)
			{
				lineJumpPoint++;
			}
		}
		if (lineJumpPoint>LINE_COUNT)
		{
			outRect.height = i - outRect.y + 5;//留一定余量
			break;
		}
	}
	//std::cout<<"origin_img4"<<std::endl;
	return 0;
}

ErrorType WaterGaugeThread::draftLocation2()
{
		//std::cout<<"origin_img1"<<std::endl;
	if (draft._lastError != Normal)
	{
		return draft._lastError;
	}
#ifdef VIEWIMAGE
	cv::imshow("_grayMat", draft._grayMat);
#endif
	GaussianBlur(draft._grayMat, draft._grayMat, cv::Size(3, 3), 0, 0);
	cv::Mat tempMat;
	draft._grayMat.copyTo(tempMat);
	int index = 0;
	int elementSize = 15;
	cv::Rect tempRect;
	bool flag = false;
	double angle = 0;
		//std::cout<<"origin_img2"<<std::endl;
	while (index<3)
	{
		cv::Mat element1 = getStructuringElement(cv::MORPH_RECT, cv::Size(elementSize, elementSize));
		dilate(tempMat, tempMat, element1);
#ifdef VIEWIMAGE
		cv::imshow("水尺定位的膨胀图1", tempMat);
#endif
		threshold(tempMat, tempMat, 125, 250, CV_THRESH_OTSU);

#ifdef VIEWIMAGE
		cv::Mat temImage;
		resize(tempMat, temImage, cv::Size(tempMat.cols / 2, tempMat.rows / 2), 0, 0, cv::INTER_LINEAR);
#endif

		std::vector<std::vector<cv::Point>> contours;
		cv::findContours(tempMat, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
		std::vector<cv::RotatedRect> box(contours.size()); //定义最小外接矩形集合
		int maxAreaIndex = -1;
		double maxArea = 0;
			//std::cout<<"origin_img3"<<std::endl;
        if (contours.empty()) {
            return InputFileError; 
	    }
		//获取矫正后的水尺的最小外接矩形下标
		for (size_t i = 0; i < contours.size(); i++)
		{
			box[i] = cv::minAreaRect(cv::Mat(contours[i]));  //计算每个轮廓最小外接矩形
			double tempArea = box[i].size.area();
			if (tempArea>maxArea)
			{
				maxArea = tempArea;
				maxAreaIndex = i;
			}
		}
		tempRect = box[maxAreaIndex].boundingRect();
		if (tempRect.width <= 25)
		{
			index++;
			if (index == 3)
			{
				return InputFileError;
			}
		}
		else
		{
			tempRect = cv::boundingRect(cv::Mat(contours[maxAreaIndex]));
			angle = box[maxAreaIndex].angle;
			//角度确定  ： 左旋和右旋的不同
			if (angle <-45)
			{
				angle = 90 + angle;
			}
			flag = true;
			break;
		}
	}


	if (flag)
	{
		cv::Mat  rectoreGray = draft._inputMat(tempRect);
#ifdef VIEWIMAGE 
		cv::imshow("截取水尺图像图", rectoreGray);
#endif
		cv::Mat  rectoreGray2;
		//矫正图像
		imageRotate(rectoreGray, rectoreGray2, angle);
#ifdef VIEWIMAGE
		cv::imshow("矫正图像", rectoreGray2);
#endif
		cv::Mat tempColorDraftMat = draft._grayMat(tempRect);
		//修正上下边缘
		int ret = findUpDownBounding(tempColorDraftMat, draft._targetRect);
		if (ret != 0 || draft._targetRect.height + 2 * elementSize < tempRect.height)
		{
			draft._targetRect.y = 0;
			draft._targetRect.height = tempRect.height;
			//std::cout << "没有找到上下边缘" << std::endl;
		}

		draft._targetRect.x = elementSize - 5;
		draft._targetRect.width = tempRect.width - 2 * elementSize + 10;
		draft._colorDraftMat = rectoreGray2(draft._targetRect);
		//std::cout<<"origin_img7"<<std::endl;
#ifdef VIEWIMAGE 
		cv::Mat  rectoreMat = rectoreGray(draft._targetRect);
		cv::imshow("扫描线裁剪的水尺图像", rectoreMat);

		cv::Mat  imgHsv2;
		imgHsv2.create(rectoreMat.rows, rectoreMat.cols, CV_8UC3);
		cv::cvtColor(rectoreMat, imgHsv2, CV_BGR2HSV);
		std::vector <cv::Mat> vecHsv2;
		cv::split(imgHsv2, vecHsv2);//分离H S I 
		cv::imshow("水尺的S分量", vecHsv2[1]);
#endif

		//能量修正
		cv::Mat  imgHsv1;
		imgHsv1.create(draft._colorDraftMat.rows, draft._colorDraftMat.cols, CV_8UC3);
		cv::cvtColor(draft._colorDraftMat, imgHsv1, CV_BGR2HSV);
		std::vector <cv::Mat> vecHsv1;
		cv::split(imgHsv1, vecHsv1);//分离H S I 
									//#ifdef VIEWIMAGE 
									//		cv::imshow("水尺的S分量", vecHsv1[1]);
									//#endif
		Sobel(vecHsv1[1], vecHsv1[1], vecHsv1[1].depth(), 0, 1);
		GaussianBlur(vecHsv1[1], vecHsv1[1], cv::Size(7, 7), 0, 0);

		std::vector<int> showHistVectS;
		int tempRows1 = draft._colorDraftMat.rows;//高
		int tempCols1 = draft._colorDraftMat.cols;//宽
#ifdef VIEWIMAGE 
		for (int i = 0; i<tempRows1; ++i)
		{
			int tempNumberS = 0;
			for (int j = 0; j<tempCols1; ++j)
			{
				//if (vecHsv1[1].at<uchar>(i, j) > 100)
				//{
				//	tempNumberS += 1;
				//}

				tempNumberS += vecHsv1[1].at<uchar>(i, j);
			}
			showHistVectS.push_back(tempNumberS);
		}

		drawHistogram(showHistVectS, "能量函数");
#endif

		for (int i = tempRows1 - 1; i>tempRows1 / 2; --i)
		{
			int tempNumberS = 0;
			for (int j = 0; j<tempCols1; ++j)
			{
				if (vecHsv1[1].at<uchar>(i, j)>100)
				{
					tempNumberS += 1;
				}
			}
			if (tempNumberS>10)
			{
				if (tempRows1 - i >5)
				{
					draft._targetRect.height = draft._targetRect.height - tempRows1 + i;
				}
				break;
			}

		}

		draft._colorDraftMat = rectoreGray2(draft._targetRect);
#ifdef VIEWIMAGE 
		cv::Rect tempRect1;
		tempRect1.x = draft._targetRect.x;
		tempRect1.y = draft._targetRect.y;
		tempRect1.width = draft._targetRect.width;
		tempRect1.height = draft._targetRect.height - 5;

		cv::Mat  img = rectoreGray(tempRect1);
		cv::imshow("重裁剪的水尺图像", img);


		cv::imshow("修正后的水尺图像", draft._colorDraftMat);
		//矫正图像
		imageRotate(draft._inputMat, draft._correctColorMat, angle);
		cv::imshow("原图矫正", draft._correctColorMat);
//std::cout<<"origin_img8"<<std::endl;
#endif
		cv::Mat  imgHsi;
		imgHsi.create(draft._colorDraftMat.rows, draft._colorDraftMat.cols, CV_8UC3);
		draft._colorDraftMat.copyTo(imgHsi);
		//cv::cvtColor(draft._colorDraftMat, imgHsi, CV_BGR2HSV);
		//vector <Mat> vecHsi;
		//cv::split(imgHsi, vecHsi);//分离H S I 
		draft._grayDraftMat.create(draft._colorDraftMat.rows, draft._colorDraftMat.cols, CV_8UC1);
		std::vector <cv::Mat> vecRGB;
		cv::split(imgHsi, vecRGB);//分离H S I 
		for (int y = 0; y < vecRGB[0].rows; ++y)
		{
			for (int x = 0; x < vecRGB[0].cols; ++x)
			{
				if (vecRGB[0].at<uchar>(y, x) == 255 && vecRGB[1].at<uchar>(y, x) == 255 && vecRGB[2].at<uchar>(y, x) == 255)
				{
					draft._grayDraftMat.at<uchar>(y, x) = 0;
				}
				else
				{
					draft._grayDraftMat.at<uchar>(y, x) = 255;
				}
			}
		}
#ifdef VIEWIMAGE 
		cv::imshow("灰度水尺图像", draft._grayDraftMat);
#endif
	}
}

ErrorType WaterGaugeThread::draftLocation()
{
	//std::cout<<"origin_img1"<<std::endl;
	if (draft._lastError != Normal)
	{
		return draft._lastError;
	}
	//找水尺的上下边界
	int ret = findUpDownBounding(draft._grayMat, draft._targetRect);
	//std::cout<<"origin_img3"<<std::endl;
	if (ret == 0)
	{
		//采用下半部分作为参考，屏蔽噪声  可优化
		cv::Rect rect(draft._targetRect.x, draft._targetRect.y + draft._targetRect.height / 2, draft._targetRect.width, draft._targetRect.height / 2);
		cv::Mat  rectoreGray = draft._grayMat(rect);
#ifdef VIEWIMAGE 
		cv::imshow("旋转矩形：", rectoreGray);
#endif
		cv::Rect tempRect;
		double angle = findRotateAngle(rectoreGray, tempRect);//若tempRect变化  里面代码需要相应更改
		//std::cout<<"origin_img5"<<std::endl;
		draft._targetRect.x = tempRect.x;
		draft._targetRect.width = tempRect.width;
		//矫正图像
		imageRotate(draft._grayMat, draft._grayCorrectMat, angle);

		//修正上下边缘
		cv::Mat element = getStructuringElement(cv::MORPH_RECT, cv::Size(13, 13));
		//进行形态学操作  ,获取最小外接矩形集合，再次进行修正
		cv::Mat  dilateMst;
		dilate(draft._grayCorrectMat, dilateMst, element);
		std::vector<std::vector<cv::Point>> contours;
		cv::findContours(dilateMst, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

		int cnt = contours.size();
		std::vector<cv::RotatedRect> box(contours.size()); //定义最小外接矩形集合

		int maxAreaIndex = 0;
		double maxArea = 0;

		//水尺是外接矩形面积最大者
		for (size_t i = 0; i < contours.size(); i++)
		{
			box[i] = cv::minAreaRect(cv::Mat(contours[i]));  //计算每个轮廓最小外接矩形
			double tempArea = box[i].size.area();
			if (tempArea>maxArea)
			{
				maxArea = tempArea;
				maxAreaIndex = i;
			}
		}

		tempRect = cv::boundingRect(cv::Mat(contours[maxAreaIndex]));
		draft._targetRect.y = tempRect.y;
		draft._targetRect.height = tempRect.height;
		draft._targetRect.x = tempRect.x;
		draft._targetRect.width = tempRect.width;
		//修正左右边缘
		cv::Mat edge, binEdge;

		Sobel(draft._grayCorrectMat, edge, draft._grayMat.depth(), 0, 1);
		//对灰度图进行滤波  
		medianBlur(edge, binEdge, 3);
		binEdge = thinImage(binEdge);
		//通过扫描线再次修正水尺位置
		scannerLine(binEdge, draft._targetRect);
                //std::cout<<"origin_img6"<<std::endl;
	}
}

double WaterGaugeThread::findRotateAngle(cv::Mat& inImg, cv::Rect& outRect)
{

	int width = inImg.cols;
	int hight = inImg.rows;

	cv::Mat element = getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
	//进行形态学操作  
	cv::Mat  dilateMst;

	dilate(inImg, dilateMst, element);

#ifdef VIEWIMAGE
	imshow("膨胀", dilateMst);
#endif

	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(dilateMst, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

	int cnt = contours.size();
	std::vector<cv::RotatedRect> box(contours.size()); //定义最小外接矩形集合

	int maxAreaIndex = 0;
	double maxArea = 0;
	//水尺对应最小外接矩形
	for (size_t i = 0; i < contours.size(); i++)
	{
		box[i] = cv::minAreaRect(cv::Mat(contours[i]));  //计算每个轮廓最小外接矩形
		double tempArea = box[i].size.area();
		if (tempArea>maxArea)
		{
			maxArea = tempArea;
			maxAreaIndex = i;
		}
	}
	double angle = box[maxAreaIndex].angle;

	//矫正图片
	imageRotate(dilateMst, dilateMst, angle);
	maxAreaIndex = 0;
	maxArea = 0;
	//获取矫正后的水尺的最小外接矩形下标
	for (size_t i = 0; i < contours.size(); i++)
	{
		box[i] = cv::minAreaRect(cv::Mat(contours[i]));  //计算每个轮廓最小外接矩形
		double tempArea = box[i].size.area();
		if (tempArea>maxArea)
		{
			maxArea = tempArea;
			maxAreaIndex = i;
		}
	}

	outRect = cv::boundingRect(cv::Mat(contours[maxAreaIndex]));
	//角度确定  ： 左旋和右旋的不同
	if (angle <-45)
	{
		return 90 + angle;
	}
	else
	{
		return angle;
	}

}

void WaterGaugeThread::imageRotate(cv::Mat& img, cv::Mat& newIm, double angle)
{
	int len = std::max(img.cols, img.rows);
	cv::Point2f pt(len / 2., len / 2.);
	cv::Mat r = getRotationMatrix2D(pt, angle, 1.0);
	warpAffine(img, newIm, r, cv::Size(len, len));
}

cv::Mat WaterGaugeThread::thinImage(cv::Mat & src, const int maxIterations)
{
	assert(src.type() == CV_8UC1);
	cv::Mat dst;
	int width = src.cols;
	int height = src.rows;
	src.copyTo(dst);
	int count = 0;  //记录迭代次数  
	while (true)
	{
		count++;
		if (maxIterations != -1 && count > maxIterations) //限制次数并且迭代次数到达  
			break;
		std::vector<uchar *> mFlag; //用于标记需要删除的点  
									//对点标记  
		for (int i = 0; i < height; ++i)
		{
			uchar * p = dst.ptr<uchar>(i);
			for (int j = 0; j < width; ++j)
			{
				//如果满足四个条件，进行标记  
				//  p9 p2 p3  
				//  p8 p1 p4  
				//  p7 p6 p5  
				uchar p1 = p[j];
				if (p1 != 1) continue;
				uchar p4 = (j == width - 1) ? 0 : *(p + j + 1);
				uchar p8 = (j == 0) ? 0 : *(p + j - 1);
				uchar p2 = (i == 0) ? 0 : *(p - dst.step + j);
				uchar p3 = (i == 0 || j == width - 1) ? 0 : *(p - dst.step + j + 1);
				uchar p9 = (i == 0 || j == 0) ? 0 : *(p - dst.step + j - 1);
				uchar p6 = (i == height - 1) ? 0 : *(p + dst.step + j);
				uchar p5 = (i == height - 1 || j == width - 1) ? 0 : *(p + dst.step + j + 1);
				uchar p7 = (i == height - 1 || j == 0) ? 0 : *(p + dst.step + j - 1);
				if ((p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) >= 2 && (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) <= 6)
				{
					int ap = 0;
					if (p2 == 0 && p3 == 1) ++ap;
					if (p3 == 0 && p4 == 1) ++ap;
					if (p4 == 0 && p5 == 1) ++ap;
					if (p5 == 0 && p6 == 1) ++ap;
					if (p6 == 0 && p7 == 1) ++ap;
					if (p7 == 0 && p8 == 1) ++ap;
					if (p8 == 0 && p9 == 1) ++ap;
					if (p9 == 0 && p2 == 1) ++ap;

					if (ap == 1 && p2 * p4 * p6 == 0 && p4 * p6 * p8 == 0)
					{
						//标记  
						mFlag.push_back(p + j);
					}
				}
			}
		}

		//将标记的点删除  
		for (std::vector<uchar *>::iterator i = mFlag.begin(); i != mFlag.end(); ++i)
		{
			**i = 0;
		}

		//直到没有点满足，算法结束  
		if (mFlag.empty())
		{
			break;
		}
		else
		{
			mFlag.clear();//将mFlag清空  
		}

		//对点标记  
		for (int i = 0; i < height; ++i)
		{
			uchar * p = dst.ptr<uchar>(i);
			for (int j = 0; j < width; ++j)
			{
				//如果满足四个条件，进行标记  
				//  p9 p2 p3  
				//  p8 p1 p4  
				//  p7 p6 p5  
				uchar p1 = p[j];
				if (p1 != 1) continue;
				uchar p4 = (j == width - 1) ? 0 : *(p + j + 1);
				uchar p8 = (j == 0) ? 0 : *(p + j - 1);
				uchar p2 = (i == 0) ? 0 : *(p - dst.step + j);
				uchar p3 = (i == 0 || j == width - 1) ? 0 : *(p - dst.step + j + 1);
				uchar p9 = (i == 0 || j == 0) ? 0 : *(p - dst.step + j - 1);
				uchar p6 = (i == height - 1) ? 0 : *(p + dst.step + j);
				uchar p5 = (i == height - 1 || j == width - 1) ? 0 : *(p + dst.step + j + 1);
				uchar p7 = (i == height - 1 || j == 0) ? 0 : *(p + dst.step + j - 1);

				if ((p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) >= 2 && (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) <= 6)
				{
					int ap = 0;
					if (p2 == 0 && p3 == 1) ++ap;
					if (p3 == 0 && p4 == 1) ++ap;
					if (p4 == 0 && p5 == 1) ++ap;
					if (p5 == 0 && p6 == 1) ++ap;
					if (p6 == 0 && p7 == 1) ++ap;
					if (p7 == 0 && p8 == 1) ++ap;
					if (p8 == 0 && p9 == 1) ++ap;
					if (p9 == 0 && p2 == 1) ++ap;

					if (ap == 1 && p2 * p4 * p8 == 0 && p2 * p6 * p8 == 0)
					{
						//标记  
						mFlag.push_back(p + j);
					}
				}
			}
		}

		//将标记的点删除  
		for (std::vector<uchar *>::iterator i = mFlag.begin(); i != mFlag.end(); ++i)
		{
			**i = 0;
		}

		//直到没有点满足，算法结束  
		if (mFlag.empty())
		{
			break;
		}
		else
		{
			mFlag.clear();//将mFlag清空  
		}
	}
	return dst;
}

int  WaterGaugeThread::scannerLine(cv::Mat & inImg, cv::Rect& ioRect)
{
	int width = inImg.cols;
	int hight = inImg.rows;
	unsigned int *lineJumpPoint = new unsigned int[width];

	for (unsigned int i = 0; i <width; ++i)
	{
		lineJumpPoint[i] = 0;
	}

	//横向扫描一幅图，统计一幅图的跳变个数
	for (unsigned int i = ioRect.x; i <ioRect.width + ioRect.x; ++i)
	{
		for (unsigned int j = 0; j< hight - 1; ++j)
		{
			if ((unsigned int)abs((inImg.at<uchar>(j, i) - inImg.at<uchar>(j + 1, i))) >200)
			{
				lineJumpPoint[i]++;
			}
		}

	}

	//寻找左边缘
	int  leftIndex = 0;
	for (int i = ioRect.x; i<ioRect.width - 3 + ioRect.x; ++i)
	{
		if (lineJumpPoint[i] >= JUMP_COUNT && lineJumpPoint[i + 1] >= JUMP_COUNT && lineJumpPoint[i + 2] >= JUMP_COUNT)
		{
			leftIndex = i;
			break;
		}

	}
	if (leftIndex == 0 || leftIndex == width - 2)
	{
		return -2;//没有找到左边缘
	}

	int rightIndex = ioRect.x + ioRect.width - 3;
	for (int i = rightIndex; i >= (leftIndex + 2); --i)
	{
		if (lineJumpPoint[i] >= JUMP_COUNT && lineJumpPoint[i - 1] >= JUMP_COUNT && lineJumpPoint[i - 2] >= JUMP_COUNT)
		{
			rightIndex = i + 2;//预留两个点
			break;
		}

	}
	if ((rightIndex <= (leftIndex + LEFT_RIGHT_COUNT)))
	{
		return -3;//没有找到右边缘
	}
	//if (ioRect.width>rightIndex - leftIndex)
	//{

	//}
	ioRect.x = leftIndex;
	ioRect.width = rightIndex - leftIndex;
	//Rect rect(leftIndex, 0, rightIndex- leftIndex, hight);
        //std::cout<<1<<std::endl;
	delete[]lineJumpPoint;
	return 0;
}

int WaterGaugeThread::getScaler(cv::Mat & inMat, cv::Rect inRect)
{
	int sizeStructuringElement = inRect.width / 4;
	if (sizeStructuringElement % 2 == 0)
	{
		sizeStructuringElement--;
	}
	//将水尺分割为左右部分,目的：消除文字
	cv::Rect leftRect;
	leftRect.x = 0;
	leftRect.y = 0;
	leftRect.width = inRect.width / 2;
	leftRect.height = inRect.height;

	cv::Rect rightRect;
	rightRect.x = inRect.width / 2;
	rightRect.y = 0;
	rightRect.width = inRect.width / 2;
	rightRect.height = inRect.height;
	cv::Mat  allMat;

	allMat.create(inRect.height, inRect.width, CV_8UC1);


	cv::Mat imageRoiRight = inMat(rightRect);
	cv::Mat imageRoiLeft = inMat(leftRect);
	//定义核  
	cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(sizeStructuringElement, sizeStructuringElement));
	//进行形态学操作  
	cv::Mat erodeRightMat;
	//进行腐蚀操作  
	dilate(imageRoiRight, erodeRightMat, element);
	// 先腐蚀再膨胀
	erode(erodeRightMat, erodeRightMat, element);
	//用右半部分消除左边部门的文字
	for (int i = 0; i<leftRect.width; ++i)
	{
		for (int j = 0; j < leftRect.height; ++j)
		{
			inMat.at<uchar>(j, i) = imageRoiLeft.at<uchar>(j, i) &(255 - erodeRightMat.at<uchar>(j, i));
		}
	}

	//消除竖线，使得全部变为横线的刻度
	//定义核  
	cv::Mat element1 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(sizeStructuringElement, 1));
	//进行形态学操作  
	cv::Mat erodeMat;
	// 先腐蚀再膨胀
	erode(inMat, erodeMat, element1, cv::Point(-1, -1));
	//对灰度图进行滤波  
	GaussianBlur(erodeMat, erodeMat, cv::Size(sizeStructuringElement, sizeStructuringElement), 0, 0);
	dilate(erodeMat, erodeMat, element1, cv::Point(-1, -1));
	Sobel(erodeMat, erodeMat, erodeMat.depth(), 0, 1);
#ifdef _DEBUG
	cv::imshow("erodeMat", erodeMat);
#endif 

	//读取刻度
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(erodeMat, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

	int cnt = contours.size();
	if (cnt == 0)
	{
		return -1;
	}
	std::vector<cv::Rect> boundRect(contours.size());  //定义外接矩形集合

	int countourHeightAverage = 0;
	//获取平均高度
	for (size_t i = 0; i < contours.size(); i++)
	{

		boundRect[i] = cv::boundingRect(cv::Mat(contours[i]));
		countourHeightAverage += boundRect[i].height;
	}
	countourHeightAverage = countourHeightAverage / cnt;

	//修正刻度
	for (size_t i = 0; i < contours.size(); i++)
	{
		//修正横向连接的刻度线 （实际为两个）
		if (boundRect[i].width >= 0.9* inRect.width)
		{
			cnt++;
		}
		//修正竖线连接的刻度线 （实际为两个）
		if (boundRect[i].height >2 * countourHeightAverage)
		{
			cnt++;
		}
		//删除小噪声误识别为刻度线，通过面积
		if (boundRect[i].width*boundRect[i].height < 0.25*inRect.width*countourHeightAverage)
		{
			cnt--;
		}
	}

#ifdef _DEBUG
	//画联通区域
	cv::Mat result1;
	erodeMat.copyTo(result1);
	for (size_t i = 0; i < contours.size(); i++)
	{
		cv::Rect r = cv::boundingRect(contours[i]);

		if (boundRect[i].width*boundRect[i].height >= 0.25*inRect.width*countourHeightAverage)
		{
			cv::rectangle(result1, r, cv::Scalar(255,0,0));
		}
	}

	cv::imshow("all regions", result1);
#endif 
	return cnt;
}
void WaterGaugeThread::drawHistogram(std::vector<int> showHistVect)
{


	cv::Mat histImg = cv::Mat::zeros(256, showHistVect.size() * 2, CV_8U);//
																  //画横纵坐标
	cv::line(histImg, cv::Point(0, 0), cv::Point(0, 256), cv::Scalar(255));

	cv::line(histImg, cv::Point(0, 256), cv::Point(showHistVect.size() * 2, 256), cv::Scalar(255));

	int maxValue = 0;
	for (int i = 0; i<showHistVect.size(); ++i)
	{
		if (maxValue < showHistVect[i])
		{
			maxValue = showHistVect[i];
		}
	}
	double scale = 1.0 * 256 / maxValue;
	for (int i = 0; i<showHistVect.size(); ++i)
	{
		cv::Rect tempRect(i * 2, 256 - showHistVect[i] * scale, 2, showHistVect[i] * scale);//
		rectangle(histImg, tempRect, cv::Scalar(255));
	}
#ifdef VIEWIMAGE
	cv::imshow(nameWindows, histImg);
#endif
}
