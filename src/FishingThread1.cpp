#include "FishingThread.h"

#include "Analysis.h"

#include "DefineColor.h"
#include <glog/logging.h>

FishingThread::FishingThread(Analysis* analysis_manager) 
    : manager(analysis_manager) {
}

void FishingThread::Run() {
    std::lock_guard<std::mutex> lk(_mutex);
    _is_run = true;
    LOG(INFO) << "FishingThread start";
    uint32_t interval = config.detect_interval() * 1000;
    std::cout << "test" << std::endl;
    while (_is_run) {
        cv::Mat origin_img = manager->GetOriginImg();
        cv::Mat segment_img = manager->GetSegmentImg();
        cv::Mat bkg_img = manager->GetForegroundImg();
        if (origin_img.empty() || segment_img.empty() || bkg_img.empty()) {
            usleep(interval);
            continue;
        }
#ifdef _DEBUG
        if (config.has_display() && config.display()) {
            std::string window_name = std::to_string(manager->Config().video_id());
            std::string origin_name = window_name + std::string("_Fishing_origin");
            std::string segment_name = window_name + std::string("_Fishing_segment");

            cv::imshow(origin_name, origin_img);
            cv::imshow(segment_name, segment_img);
            cv::waitKey(1);
        }
#endif
        std::vector<cv::Rect> people_rect;
        if (FishingRodEstimate(origin_img, segment_img, people_rect))
            SetAlarm(true, people_rect);
        else
            SetAlarm(false, people_rect);
        usleep(interval);
    }
    LOG(INFO) << "FishingThread end";
    _is_run = false;
    CallStop();
}

AnalysisAlarm FishingThread::GetAlarm() {
    std::lock_guard<std::mutex> lk(alarm_mutex);
    AnalysisAlarm res = alarm;
    alarm.set_is_active(false);
    return res;
}

void FishingThread::SetAlarm(bool is_active, const std::vector<cv::Rect>& people_rect) {
    std::lock_guard<std::mutex> lk(alarm_mutex);
    alarm.set_is_active(is_active);
    alarm.clear_rects();
    if (!is_active)
        return;
    for (const cv::Rect& rect : people_rect) {
        AnalysisRect* newrect = alarm.add_rects();
        newrect->set_x(rect.x);
        newrect->set_y(rect.y);
        newrect->set_width(rect.width);
        newrect->set_height(rect.height);
    }
}


void FishingThread::GetEdge(cv::Mat srcImg, cv::Mat &edgeMat)
{//getting edge 
	cv::Mat grayImage, enhanceImg, erzhihua;
	cv::cvtColor(srcImg, grayImage, CV_RGB2GRAY);
	cv::threshold(grayImage, grayImage, 10, 255, CV_THRESH_BINARY);
	bitwise_not(grayImage, edgeMat);

}


void FishingThread::TargetDectRect(cv::Mat segmentMat,const cv::Scalar& segmentColor, std::vector<cv::Point> &targetPoint,std::vector<cv::Rect> &targetRect)
{
	cv::Mat segImg;
	inRange(segmentMat, segmentColor,segmentColor, segImg);//寻找颜色
	threshold(segImg, segImg, 200, 255, CV_THRESH_OTSU);
		
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	std::vector<cv::Point> contourRect;
	cv::findContours(segImg, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE, cv::Point());
	for (size_t i = 0; i < contours.size(); i++)
	{
		if (contours[i].size() > 65)
		{
			cv::RotatedRect rect = minAreaRect(contours[i]);
			targetPoint.push_back(rect.center);
			targetRect.push_back(rect.boundingRect());
		}
	}
}

bool FishingThread::GetDetectRegion(cv::Mat &fishingDetectRegion,int sceneMold)
{
	cv::Mat edgeMat= cv::Mat::zeros(fishingDetectRegion.rows, fishingDetectRegion.cols, CV_8UC1);
	GetEdge(fishingDetectRegion, edgeMat);
	
	//去除水平垂直线
	cv::Mat edgeOutMat;
	DeleteHVLine(edgeMat, edgeOutMat);
	//去除小点等
	cv::Mat romoveNoiseMat/* = cv::Mat::zeros(edgeMat.rows, edgeMat.cols, CV_8UC1)*/;
	IplImage inputImg = edgeOutMat;
	DeleteSmallArea(&inputImg, romoveNoiseMat);
	//直线拟合
	IplImage houghImg = romoveNoiseMat;
	std::vector<cv::Point> binWhitePoint;
	bool isFishing = LineFitting(&houghImg, binWhitePoint, sceneMold);
	return isFishing;
}

bool FishingThread::FishingRodEstimate(cv::Mat srcImg, cv::Mat segmentMat, std::vector<cv::Rect>& people_rect)
{
	std::vector<cv::Point> peopleCenterPoint;
	std::vector<cv::Rect> peopleSegmentRect;
	std::vector<cv::Point> riverCenterPoint;
	std::vector<cv::Rect> riverSegmentRect;
	TargetDectRect(segmentMat, PEOPLE_COLOR, peopleCenterPoint, peopleSegmentRect);
	TargetDectRect(segmentMat, WATER_COLOR , riverCenterPoint, riverSegmentRect);

	double widthRatio = (double)srcImg.cols / segmentMat.cols;
	double heightRatio = (double)srcImg.rows / segmentMat.rows;
    people_rect = peopleSegmentRect;
    for (size_t i = 0; i < people_rect.size(); ++i) {
        people_rect[i].x = people_rect[i].x * widthRatio;
        people_rect[i].y = people_rect[i].y * heightRatio;
        people_rect[i].width = people_rect[i].width * widthRatio;
        people_rect[i].height = people_rect[i].height * heightRatio;
    }

	if (peopleCenterPoint.size() == 0|| riverCenterPoint.size() == 0 )
		return false;

	cv::Point riverLocation(0, 0);
	int maxArea = 0;
	for (size_t j = 0; j < riverCenterPoint.size(); ++j)
	{
		if (riverSegmentRect[j].height * riverSegmentRect[j].width > maxArea)
		{
			maxArea = riverSegmentRect[j].height * riverSegmentRect[j].width;
			riverLocation.x = riverCenterPoint[j].x;
			riverLocation.y = riverCenterPoint[j].y;
		}
	}
	if (riverLocation.x == 0 || riverLocation.y == 0)
		return false;

    bool isFinshing = false;
	for (size_t i = 0; i < peopleCenterPoint.size(); ++i)
	{
		cv::Rect fishingRodRect;
		cv::Rect peopleRealRect;
		cv::Rect riverRealRect;

		peopleRealRect.x = peopleSegmentRect[i].x * widthRatio;
		peopleRealRect.y = peopleSegmentRect[i].y * heightRatio;
		peopleRealRect.width = peopleSegmentRect[i].width * widthRatio;
		peopleRealRect.height = peopleSegmentRect[i].height * heightRatio;

		if (peopleRealRect.height > peopleRealRect.width)
			return false;

		if (peopleCenterPoint[i].x > riverLocation.x)//河左人右
		{
			fishingRodRect.x = peopleRealRect.x - peopleRealRect.width * 2;
			if (fishingRodRect.x <= 0)
				fishingRodRect.x = abs(peopleSegmentRect[i].x - riverLocation.x)*widthRatio;
			fishingRodRect.y = peopleRealRect.y *0.9 ;
			fishingRodRect.width = peopleRealRect.width * 2 - 2;

			fishingRodRect.height = peopleRealRect.height * 1.5 ;
			if(fishingRodRect.height+ fishingRodRect.y>srcImg.rows)
				fishingRodRect.height= srcImg.rows - fishingRodRect.y - 5;
			cv::Mat fishingRodDetectRegion = srcImg(fishingRodRect);

			rectangle(srcImg, peopleRealRect, cv::Scalar(255, 0, 255));

			isFinshing = GetDetectRegion(fishingRodDetectRegion, 0);
			if (isFinshing == true)
				return true;
			
		}

		if (peopleCenterPoint[i].x < riverLocation.x)//人左河右
		{
			fishingRodRect.x = peopleRealRect.x;
			fishingRodRect.y = peopleRealRect.y*0.9;

			fishingRodRect.width = peopleRealRect.width * 2 - 2;
			if (fishingRodRect.x + fishingRodRect.width > srcImg.cols)
				fishingRodRect.width = abs(peopleSegmentRect[i].x - riverLocation.x)*widthRatio;
			fishingRodRect.height = peopleRealRect.height * 1.5 ;
			if (fishingRodRect.height + fishingRodRect.y > srcImg.rows)
				fishingRodRect.height = srcImg.rows - fishingRodRect.y - 5;
			cv::Mat fishingRodDetectRegion = srcImg(fishingRodRect);

			rectangle(srcImg, peopleRealRect, cv::Scalar(0, 0, 255));
			isFinshing = GetDetectRegion(fishingRodDetectRegion, 1);
			if (isFinshing == true)
				return true;

		}
    }
    return false;
}

void FishingThread::RemoveSmallRegion(cv::Mat &Src, cv::Mat &Dst, size_t AreaLimit, int CheckMode, int NeihborMode)
{//CheckMode: 0代表去除黑区域，1代表去除白区域; NeihborMode：0代表4邻域，1代表8邻域;
	int RemoveCount = 0;
	cv::Mat PointLabel = cv::Mat::zeros(Src.size(), CV_8UC1);
	if (CheckMode == 1)		//去除小连通区域的白色点  
	{
		for (int i = 0; i < Src.rows; i++)
		{
			for (int j = 0; j < Src.cols; j++)
			{
				if (Src.at<uchar>(i, j) < 10)
				{
					PointLabel.at<uchar>(i, j) = 3;  //将背景黑色点标记为合格，像素为3  
				}
			}
		}
	}
	else//去除孔洞，黑色点像素  
	{
		for (int i = 0; i < Src.rows; i++)
		{
			for (int j = 0; j < Src.cols; j++)
			{
				if (Src.at<uchar>(i, j) > 10)
				{
					PointLabel.at<uchar>(i, j) = 3;//如果原图是白色区域，标记为合格，像素为3  
				}
			}
		}
	}
	std::vector<cv::Point2i>NeihborPos;//将邻域压进容器  
	NeihborPos.push_back(cv::Point2i(-1, 0));
	NeihborPos.push_back(cv::Point2i(1, 0));
	NeihborPos.push_back(cv::Point2i(0, -1));
	NeihborPos.push_back(cv::Point2i(0, 1));
	if (NeihborMode == 1)
	{
		// "Neighbor mode: 8邻域."
		NeihborPos.push_back(cv::Point2i(-1, -1));
		NeihborPos.push_back(cv::Point2i(-1, 1));
		NeihborPos.push_back(cv::Point2i(1, -1));
		NeihborPos.push_back(cv::Point2i(1, 1));
	}
	else
	{
		//"Neighbor mode: 4邻域."
	}
	int NeihborCount = 4 + 4 * NeihborMode;
	int CurrX = 0, CurrY = 0;
	//开始检测  
	for (int i = 0; i < Src.rows; i++)
	{
		for (int j = 0; j < Src.cols; j++)
		{
			if (PointLabel.at<uchar>(i, j) == 0)//标签图像像素点为0，表示还未检查的不合格点  
			{   //开始检查  
				std::vector<cv::Point2i> GrowBuffer;//记录检查像素点的个数  
				GrowBuffer.push_back(cv::Point2i(j, i));
				PointLabel.at<uchar>(i, j) = 1;//标记为正在检查  
				int CheckResult = 0;
				for (size_t z = 0; z < GrowBuffer.size(); z++)
				{
					for (int q = 0; q < NeihborCount; q++)
					{
						CurrX = GrowBuffer.at(z).x + NeihborPos.at(q).x;
						CurrY = GrowBuffer.at(z).y + NeihborPos.at(q).y;
						if (CurrX >= 0 && CurrX<Src.cols&&CurrY >= 0 && CurrY<Src.rows)  //防止越界    
						{
							if (PointLabel.at<uchar>(CurrY, CurrX) == 0)
							{
								GrowBuffer.push_back(cv::Point2i(CurrX, CurrY));  //邻域点加入buffer    
								PointLabel.at<uchar>(CurrY, CurrX) = 1;       //更新邻域点的检查标签，避免重复检查    
							}
						}
					}
				}
				if (GrowBuffer.size() > AreaLimit) //判断结果（是否超出限定的大小），1为未超出，2为超出    
					CheckResult = 2;
				else
				{
					CheckResult = 1;
					RemoveCount++;              //记录有多少区域被去除  
				}
				for (size_t z = 0; z < GrowBuffer.size(); z++)
				{
					CurrX = GrowBuffer.at(z).x;
					CurrY = GrowBuffer.at(z).y;
					PointLabel.at<uchar>(CurrY, CurrX) += CheckResult;//标记不合格的像素点，像素值为2  
				}
			}
		}
	}
	CheckMode = 255 * (1 - CheckMode);
	//开始反转面积过小的区域    
	for (int i = 0; i < Src.rows; ++i)
	{
		for (int j = 0; j < Src.cols; ++j)
		{
			if (PointLabel.at<uchar>(i, j) == 2)
			{
				Dst.at<uchar>(i, j) = CheckMode;
			}
			else if (PointLabel.at<uchar>(i, j) == 3)
			{
				Dst.at<uchar>(i, j) = Src.at<uchar>(i, j);
			}
		}
	}
}

void FishingThread::DeleteSmallArea(IplImage* inputImage, cv::Mat &outputImage)
{//判断图是否具有白色点，没有则返回原图
	CvSeq* contour = NULL;
	double minarea = 50.0;
	double tmparea = 0.0;
	uchar *pp;
	CvMemStorage* storage = cvCreateMemStorage(0);
	IplImage* img_Clone = inputImage;
	cvFindContours(img_Clone, storage, &contour, sizeof(CvContour), CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE, cvPoint(0, 0));
	//开始遍历轮廓树         
	CvRect rect;
	cvDrawContours(img_Clone, contour, CV_RGB(255, 255, 255), CV_RGB(255, 255, 255), 1, CV_FILLED, 8, cvPoint(0, 0));
	while (contour)
	{
		tmparea = fabs(cvContourArea(contour));
		rect = cvBoundingRect(contour, 0);
		if (tmparea < 10)
		{
			//当连通域的中心点为黑色时，而且面积较小则用白色进行填充
			pp = (uchar*)(img_Clone->imageData + img_Clone->widthStep*(rect.y + rect.height / 2) + rect.x + rect.width / 2);
			if (pp[0] == 0)
			{
				for (int y = rect.y; y <= rect.y + rect.height; y++)
				{
					for (int x = rect.x; x <= rect.x + rect.width; x++)
					{
						pp = (uchar*)(img_Clone->imageData + img_Clone->widthStep*y + x);
						if (pp[0] == 0)
						{
							pp[0] = 255;
						}
					}
				}
			}
		}
		contour = contour->h_next;
	}

	int countPoint = 0;
	std::vector<cv::Point> binWhitePoint;
	GetBinWhitePoint(img_Clone, binWhitePoint, countPoint);
	if (countPoint < 10)
		outputImage = cv::cvarrToMat(inputImage,false);
	else
		outputImage = cv::cvarrToMat(img_Clone, false);
	//cvReleaseImage(&img_Clone);
	cvReleaseMemStorage(&storage);
}


void FishingThread::GetBinWhitePoint(IplImage* inputImage, std::vector<cv::Point>& binWhitePoint, int &countPint)
{//返回一个vector容器
	cv::Mat src = cv::cvarrToMat(inputImage, false);
	if (src.rows < 11 || src.cols < 11 || src.empty())
	{
		return;
	}
	for (int i = 10; i < src.rows-10; i++)
	{
		const uchar* pixelPtr = src.ptr<uchar>(i);
		for (int j = 10; j < src.cols - 10; j++)
		{
			if (pixelPtr[j] > 0)
			{
				binWhitePoint.push_back(cv::Point(j, i));
				++countPint;
			}
		}
	}
}

void FishingThread::DeleteHVLine(cv::Mat inputMat, cv::Mat& outMat)
{
	cv::Mat horizontalMat = inputMat.clone();
	cv::Mat verticalMat = inputMat.clone();
	int horizontalKernel = horizontalMat.cols / 30;
	int verticalKernel = verticalMat.rows / 30;

	cv::Mat horizontalLine = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(horizontalKernel, 1));
	cv::Mat verticalLine = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, verticalKernel), cv::Point(-1, -1));

	erode(horizontalMat, horizontalMat, horizontalLine, cv::Point(-1, -1));
	dilate(horizontalMat, horizontalMat, horizontalLine, cv::Point(-1, -1));

	erode(verticalMat, verticalMat, verticalLine, cv::Point(-1, -1));
	dilate(verticalMat, verticalMat, verticalLine, cv::Point(-1, -1));

	bitwise_not(horizontalMat, horizontalMat);//图像进行反转
	bitwise_not(verticalMat, verticalMat);
	cv::Mat tempMat;
	bitwise_and(inputMat, horizontalMat, tempMat);
	bitwise_and(tempMat, verticalMat, outMat);

}

bool FishingThread::LineFitting(IplImage* inputImage, std::vector<cv::Point>& binWhitePoint,int sceneMold)
{
	int countpoint = 0;
	GetBinWhitePoint(inputImage, binWhitePoint, countpoint);
	cv::Mat dst = cv::cvarrToMat(inputImage);
	cv::Mat blankImage = cv::Mat::zeros(dst.rows, dst.cols, CV_8UC1);
	//将拟合点绘制到空白图上
	for (size_t i = 0; i < binWhitePoint.size(); i++)
	{
		cv::circle(blankImage, binWhitePoint[i], 5, cv::Scalar(0, 0, 255), 1, 8, 0);
	}
	cv::Vec4f line_para;
	if (binWhitePoint.size() == 0)
		return false;
	cv::fitLine(binWhitePoint, line_para, cv::DIST_HUBER, 0, 1e-2, 1e-2);

	//最小二乘拟合计算直线的倾角
	int pointCount = binWhitePoint.size();
	if (pointCount > 0)
	{
		int xCount = 0;
		int yCount = 0;
		int xyCount = 0;
		int xxCount = 0;
		for (int i = 0; i< pointCount; i++)
		{
			xCount += binWhitePoint[i].x;
			yCount += binWhitePoint[i].y;
			xyCount += binWhitePoint[i].x * binWhitePoint[i].y;
			xxCount += binWhitePoint[i].x * binWhitePoint[i].x;
		}
		double k = (double)(pointCount * xyCount - xCount * yCount) / (double)(pointCount * xxCount - xCount * xCount);
		double sinValue = -k / (sqrt(1 + k * k));
		double radian = asin(sinValue);
		double pi = 3.1415926535;
		double angle = radian * 180.0 / pi;

		//直线用 ρ=xcosθ+ysinθ 来表示， 那么 θ=arctank+π/2
		double cosQ = line_para[0];
		double sinQ = line_para[1];
		double X0 = line_para[2];
		double Y0 = line_para[3];
		double phi = atan2(sinQ, cosQ) + CV_PI / 2;
		double rho = Y0*cosQ - X0*sinQ;

		if (phi < CV_PI / 4 || phi>3 * CV_PI / 4)
		{
			cv::Point pt1(rho / cos(phi), 0);
			cv::Point pt2((rho - blankImage.rows * sin(phi)) / cos(phi), blankImage.rows);
			cv::line(blankImage, pt1, pt2, cv::Scalar(255, 255, 255), 1);
		}
		else
		{
			cv::Point pt1(0, rho / sin(phi));
			cv::Point pt2(blankImage.cols, (rho - blankImage.cols * cos(phi)) / sin(phi));
			cv::line(blankImage, pt1, pt2, cv::Scalar(255, 255, 255), 1);
		}

#ifdef _DEBUG
        if (config.has_display() && config.display()) {
            std::string window_name = std::to_string(manager->Config().video_id());
            std::string line_name = window_name + std::string("_Fishing_line");
		    imshow(line_name, blankImage);
        }
#endif
		if (sceneMold == 1)
		{
			if (angle > 10&&angle<70)
			{
				return true;
			}
			else
				return false;
		}
		else
		{
			if (angle < -10 && angle> -50)
			{
				return true;
			}
			else
				return false;
		}
		
		if (sceneMold == 1)
		{
			if (angle > 10&&angle<70)
			{
				return true;
			}
			else
			{
				return false;
			}
		}
		else
		{
			if (angle < -10 && angle> -50)
			{
				return true;
			}
			else
			{
				return false;
			}
		}
	}
    return false;
}
