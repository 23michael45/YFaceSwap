#pragma once

#include <opencv2/core/core.hpp>
#include <vector>
#include <string>
#include <memory>

#include <dlib/image_processing/frontal_face_detector.h>
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






	dlib::frontal_face_detector m_dlibFaceDetector;
	std::vector<dlib::rectangle> m_dlibFaces;
};

