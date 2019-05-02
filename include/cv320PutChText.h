#ifndef OPENCV_320_PUTCHINESETEXT_H
#define OPENCV_320_PUTCHINESETEXT_H

#include <iostream>
#include "opencv2/opencv.hpp"
#include <ft2build.h>
#include FT_FREETYPE_H 

#include <string>
#include <assert.h>  
#include <locale.h>  
#include <ctype.h>  

class cv320PutChText
{
public:
	cv320PutChText() {}
	cv320PutChText(const char *freeType);
	~cv320PutChText();
    int Loadttc(const char* ttc_path);
    void release();
	void getFont(int *type, CvScalar *size = NULL, bool *underline = NULL, float *diaphaneity = NULL);
	void setFont(int *type, CvScalar *size = NULL, bool *underline = NULL, float *diaphaneity = NULL);
	void restoreFont(const int frontSize);
	int putText(cv::Mat &frame, std::string& text, CvPoint pos, CvScalar color,const int frontSize);
    int putText(cv::Mat &frame, const char    *text, CvPoint pos, const int frontSize);
	int putText(cv::Mat &frame, const char    *text, CvPoint pos, CvScalar color, const int frontSize);
	int putText(cv::Mat &frame, const wchar_t *text, CvPoint pos, CvScalar color, const int frontSize);

    static std::wstring stows(const std::string& s);
private:
	void putWChar(cv::Mat&frame, wchar_t wc, CvPoint &pos, CvScalar color);
private:
	FT_Library  m_library;
	FT_Face     m_face;
	int         m_fontType;
	CvScalar    m_fontSize;
	bool        m_fontUnderline;
	float       m_fontDiaphaneity;
};

#endif
