#include "FaceDetector.h"

#include <opencv2/core/core.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>

FaceDetector::FaceDetector(const std::string cascadeFilePath)
{

    m_faceCascade = std::make_unique<cv::CascadeClassifier>(cascadeFilePath);
    if (m_faceCascade->empty())
    {
        std::cerr << "Error loading cascade file " << cascadeFilePath << std::endl << 
            "Make sure the file exists" << std::endl;
        exit(-1);
    }


}

FaceDetector::~FaceDetector()
{

}


std::vector<cv::Rect> FaceDetector::faces()
{
    std::vector<cv::Rect> faces;
    for (const auto& face : m_facesRects)
    {
        faces.push_back(cv::Rect(face.x * m_ratio.x, face.y * m_ratio.y, face.width * m_ratio.x, face.height * m_ratio.y));
    }
    return faces;
}

void FaceDetector::detect(cv::Mat img)
{
	
	m_originalFrameSize.width = (int)img.cols;
	m_originalFrameSize.height = (int)img.rows;
	m_downscaledFrameSize.width = m_downscaledFrameWidth;
	m_downscaledFrameSize.height = (m_downscaledFrameSize.width * m_originalFrameSize.height) / m_originalFrameSize.width;

	m_ratio.x = (float)m_originalFrameSize.width / m_downscaledFrameSize.width;
	m_ratio.y = (float)m_originalFrameSize.height / m_downscaledFrameSize.height;


	cv::resize(img, m_downscaledFrame, m_downscaledFrameSize);


    // Minimum face size is 1/5th of screen height
    // Maximum face size is 2/3rds of screen height
    m_faceCascade->detectMultiScale(m_downscaledFrame, m_facesRects, 1.1, 3, 0,
        cv::Size(m_downscaledFrame.rows / 5, m_downscaledFrame.rows / 5),
        cv::Size(m_downscaledFrame.rows * 2 / 3, m_downscaledFrame.rows * 2 / 3));


    // Get face templates
    m_faceTemplates.clear();
    for (auto face : m_facesRects)
    {
        face.width /= 2;
        face.height /= 2;
        face.x += face.width / 2;
        face.y += face.height / 2;
        m_faceTemplates.push_back(m_downscaledFrame(face).clone());
    }

    // Get face ROIs
    m_faceRois.clear();
    for (const auto& face : m_facesRects)
    {
        m_faceRois.push_back(doubleRectSize(face, m_downscaledFrameSize));
    }

    // Initialize template matching timers
    m_tmRunningInRoi.clear();
    m_tmStartTime.clear();
    m_tmEndTime.clear();
    m_tmRunningInRoi.resize(m_facesRects.size(), false);
    m_tmStartTime.resize(m_facesRects.size());
    m_tmEndTime.resize(m_facesRects.size());


}



cv::Rect FaceDetector::doubleRectSize(const cv::Rect &inputRect, const cv::Size &frameSize)
{
    cv::Rect outputRect;
    // Double rect size
    outputRect.width = inputRect.width * 2;
    outputRect.height = inputRect.height * 2;

    // Center rect around original center
    outputRect.x = inputRect.x - inputRect.width / 2;
    outputRect.y = inputRect.y - inputRect.height / 2;

    // Handle edge cases
    if (outputRect.x < 0) {
        outputRect.width += outputRect.x;
        outputRect.x = 0;
    }
    if (outputRect.y < 0) {
        outputRect.height += outputRect.y;
        outputRect.y = 0;
    }

    if (outputRect.x + outputRect.width > frameSize.width) {
        outputRect.width = frameSize.width - outputRect.x;
    }
    if (outputRect.y + outputRect.height > frameSize.height) {
        outputRect.height = frameSize.height - outputRect.y;
    }

    return outputRect;
}
