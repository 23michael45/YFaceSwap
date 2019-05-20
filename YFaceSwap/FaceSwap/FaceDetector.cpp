#include "FaceDetector.h"
#include <opencv2/core/core.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include <memory>

#include <dlib/image_processing.h>
#include <dlib/opencv.h>

FaceDetector::FaceDetector(const std::string cascadeFilePath)
{

    //m_faceCascade = std::make_shared<cv::CascadeClassifier>(cascadeFilePath);
    //if (m_faceCascade->empty())
    //{
    //    std::cerr << "Error loading cascade file " << cascadeFilePath << std::endl << 
    //        "Make sure the file exists" << std::endl;
    //    exit(-1);
    //}

	m_dlibFaceDetector = dlib::get_frontal_face_detector();

	m_ratio.x = 1;
	m_ratio.y = 1;
}

FaceDetector::~FaceDetector()
{

}


std::vector<cv::Rect> FaceDetector::faces()
{
    std::vector<cv::Rect> faces;
    //for (const auto& face : m_facesRects)
    //{
    //    faces.push_back(cv::Rect(face.x * m_ratio.x, face.y * m_ratio.y, face.width * m_ratio.x, face.height * m_ratio.y));
    //}
	for (const auto& face :	m_dlibFaces)
	{

		long l = face.left();
		long t = face.top();
		long w = face.width();
		long h = face.height();
		
		faces.push_back(cv::Rect(l * m_ratio.x,t * m_ratio.y,w * m_ratio.x,h * m_ratio.y));
	}
    return faces;
}

void FaceDetector::detect(cv::Mat img)
{
	
	/*m_originalFrameSize.width = (int)img.cols;
	m_originalFrameSize.height = (int)img.rows;
	m_downscaledFrameSize.width = m_downscaledFrameWidth;
	m_downscaledFrameSize.height = (m_downscaledFrameSize.width * m_originalFrameSize.height) / m_originalFrameSize.width;

	m_ratio.x = (float)m_originalFrameSize.width / m_downscaledFrameSize.width;
	m_ratio.y = (float)m_originalFrameSize.height / m_downscaledFrameSize.height;


	cv::resize(img, m_downscaledFrame, m_downscaledFrameSize);*/


 //   // Minimum face size is 1/10th of screen height
 //   // Maximum face size is 2/3rds of screen height
 //   m_faceCascade->detectMultiScale(m_downscaledFrame, m_facesRects, 1.1, 3, 0,
 //       cv::Size(m_downscaledFrame.cols / 50, m_downscaledFrame.rows / 50),
 //       cv::Size(m_downscaledFrame.cols * 2 / 3, m_downscaledFrame.rows * 2 / 3));

	//dlib::cv_image<dlib::bgr_pixel> dlib_img = img;

	//dlib::cv_image<dlib::bgr_pixel> dlib_img = m_downscaledFrame;

	cv::Mat gray;
	cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
	dlib::cv_image<uchar> dlib_img = gray;


	m_dlibFaces = m_dlibFaceDetector(dlib_img);

}


