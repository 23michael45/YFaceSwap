#pragma once

#include <opencv2/core/core.hpp>
#include <vector>
#include <string>
#include <memory>

namespace cv
{
    class VideoCapture;
    class CascadeClassifier;
}



class FaceDetector
{
public:
    /*
     * Initializes detector with cascade file, initializes camera with camera index and sets number of faces to track
     */
    FaceDetector(const std::string cascadeFilePath);
    ~FaceDetector();

	void detect(cv::Mat img);
    std::vector<cv::Rect> faces();

private:

    /* Returns double inputRect size centered around the same point */
    static cv::Rect doubleRectSize(const cv::Rect &rect, const cv::Size &frameSize);

    /*
     * Private members
     */

    /*
     * Video capture object used for retrieving camera frames
     */
    std::shared_ptr<cv::VideoCapture> m_camera;

    /*
     * Cascade classifier object used for detecting faces in frames
     */
    std::shared_ptr<cv::CascadeClassifier> m_faceCascade;

    /*
     * Downscaled camera frame. Downscaling speeds up detection 
     */
    cv::Mat m_downscaledFrame;
    
    /*
     * Width of downscaled camera frame. Height is calculated to preserve aspect ratio
     */
    static const int m_downscaledFrameWidth = 256;

    /*
     * Vector of rectangles representing faces in camera frame
     */
    std::vector<cv::Rect> m_facesRects;

    /*
     * Vector of vector of faces. Used in tracking. One vector per detected face
     */
    std::vector<cv::Rect> m_tmpFacesRect;

    std::vector<bool>                       m_tmRunningInRoi;
    std::vector<long long>                  m_tmStartTime;
    std::vector<long long>                  m_tmEndTime;

    std::vector<cv::Point2f>                m_facePositions;
    std::vector<cv::Mat>                    m_faceTemplates;
    std::vector<cv::Rect>                   m_faceRois;

    cv::Mat                                 m_matchingResult;

    cv::Size                                m_downscaledFrameSize;
    cv::Size                                m_originalFrameSize;
    cv::Point2f                             m_ratio;


    size_t m_numFaces = 0;

    const double                            m_tmMaxDuration = 2.0;



};

