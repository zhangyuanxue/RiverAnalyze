#ifndef RIVER_ANALYSIS_WATERGAUGE_THREAD_H
#define RIVER_ANALYSIS_WATERGAUGE_THREAD_H

#include "RiverThread.h"
#include <opencv2/opencv.hpp>
enum DraftType { RedDraft = 0, BlueDraft };
enum ErrorType { Normal = 0, InputFileError, LocationDetectError };
struct WaterGaugeData
{
	cv::Mat _inputMat;//输入的彩色图原图
	cv::Mat _segmentMat;//输入分割
	cv::Mat _correctColorMat;//矫正彩色度图
	cv::Mat _grayMat;//灰度图
	cv::Mat _grayCorrectMat;//矫正灰度图
	cv::Mat _grayDraftMat;//矫正后水尺灰度图
	cv::Mat _colorDraftMat;//矫正后水尺灰度图
	cv::Rect _targetRect;//水尺目标在_grayCorrectMat的区域
	ErrorType _lastError;
	cv::Rect _tempRect;
        DraftType DraftColor;
};

class Analysis;
class WaterGaugeThread : public RiverThread {
public:
    WaterGaugeThread(Analysis* analysis_manager);
    ~WaterGaugeThread() {}

    void Run();
   
    AnalysisAlarm GetAlarm();
    void SetAlarm(float num);

private:
    ErrorType CWaterLevelMeasure(cv::Mat inputMatSrc, cv::Mat segmentMat, int &sraftNumber);
	ErrorType getWaterLevelScaler(int &sraftNumber);
	ErrorType WaterDraftMeasure(cv::Mat DivisionInput);
	ErrorType draftLocation();
	ErrorType draftLocation2();
	ErrorType preDealWithImage(DraftType type);
	int findUpDownBounding(cv::Mat& inImg, cv::Rect& outRect);
	double findRotateAngle(cv::Mat& inImg, cv::Rect& outRect);
	void imageRotate(cv::Mat& img, cv::Mat& newIm, double angle);
	cv::Mat thinImage(cv::Mat & src, const int maxIterations = -1);
	int  scannerLine(cv::Mat& inImg, cv::Rect& ioRect);
	int getScaler(cv::Mat & inMat, cv::Rect inRect);
	void drawHistogram(std::vector<int> showHistVect);
       
private:
    Analysis* manager;
    WaterGaugeData draft;
};

#endif
