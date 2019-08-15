#pragma once

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <iostream>

#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
//#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include "App/ini.h"
//#include <dlib/gui_widgets.h>
#define LEFT_FACE_CONTOUR_1 1
#define LEFT_FACE_CONTOUR_2 2
#define LEFT_FACE_CONTOUR_3 3
#define LEFT_FACE_CONTOUR_4 4
#define LEFT_FACE_CONTOUR_5 5
#define LEFT_FACE_CONTOUR_6 6
#define LEFT_FACE_CONTOUR_7 7
#define LEFT_FACE_CONTOUR_8 8
#define MIDDLE_CHIN 9
#define RIGHT_FACE_CONTOUR_1 17
#define RIGHT_FACE_CONTOUR_2 16
#define RIGHT_FACE_CONTOUR_3 15
#define RIGHT_FACE_CONTOUR_4 14
#define RIGHT_FACE_CONTOUR_5 13
#define RIGHT_FACE_CONTOUR_6 12
#define RIGHT_FACE_CONTOUR_7 11
#define RIGHT_FACE_CONTOUR_8 10
#define LEFT_EYEBROW_1 18
#define LEFT_EYEBROW_2 19
#define LEFT_EYEBROW_3 20
#define LEFT_EYEBROW_4 21
#define LEFT_EYEBROW_5 22
#define RIGHT_EYEBROW_1 27
#define RIGHT_EYEBROW_2 26
#define RIGHT_EYEBROW_3 25
#define RIGHT_EYEBROW_4 24
#define RIGHT_EYEBROW_5 23
#define LEFT_EYE_1 37
#define LEFT_EYE_2 38
#define LEFT_EYE_3 39
#define LEFT_EYE_4 40
#define LEFT_EYE_5 41
#define LEFT_EYE_6 42
#define RIGHT_EYE_1 46
#define RIGHT_EYE_2 45
#define RIGHT_EYE_3 44
#define RIGHT_EYE_4 43
#define RIGHT_EYE_5 48
#define RIGHT_EYE_6 47
#define  MIDDLE_NOSE_1 28
#define  MIDDLE_NOSE_2 29
#define  MIDDLE_NOSE_3 30
#define  MIDDLE_NOSE_4 31
#define  MIDDLE_NOSE_5 34
#define  LEFT_NOSE_1 32
#define  LEFT_NOSE_2 33
#define  RIGHT_NOSE_1 36
#define  RIGHT_NOSE_2 35
#define OUT_LEFT_MOUTH 49
#define OUT_RIGHT_MOUTH 55
#define OUT_UP_MOUTH 52
#define OUT_DOWN_MOUTH 58
#define IN_LEFT_MOUTH 61
#define IN_RIGHT_MOUTH 65
#define IN_UP_MOUTH 63
#define IN_DOWN_MOUTH 67
#define OUT_LEFT_UP_MOUTH_1 50
#define OUT_LEFT_UP_MOUTH_2 51
#define OUT_RIGHT_UP_MOUTH_1 54
#define OUT_RIGHT_UP_MOUTH_2 53
#define OUT_LEFT_DOWN_MOUTH_1 60
#define OUT_LEFT_DOWN_MOUTH_2 59
#define OUT_RIGHT_DOWN_MOUTH_1 56
#define OUT_RIGHT_DOWN_MOUTH_2 57
#define IN_LEFT_UP_MOUTH 62
#define IN_RIGHT_UP_MOUTH 64
#define IN_LEFT_DOWN_MOUTH 68
#define IN_RIGHT_DOWN_MOUTH 66
#define FACE_GET(keypoint) (keypoint - 1)





class FaceExchanger
{
public:
    // Initialize face swapped with landmarks
    FaceExchanger(const std::string landmarks_path, mINI::INIStructure& iniFile);
    ~FaceExchanger();

    //Swaps faces in rects on frame
    void swapFaces(cv::Mat &src,cv::Mat &dst, cv::Rect &rect_src, cv::Rect &rect_dst,bool blur = true);

private:
	// Beauty face
	cv::Mat FaceBeauty(cv::Mat &src);

    // Returns minimal Mat containing both faces
    cv::Mat getMinFrame(const cv::Mat &frame, cv::Rect &rect_src, cv::Rect &rect_dst);

    // Finds facial landmarks on faces and extracts the useful points
	void getFacePoints(const cv::Mat &src, const cv::Mat &dst);

    // Calculates transformation matrices based on points extracted by getFacePoints
    void getTransformationMatrices();

    // Creates masks for faces based on the points extracted in getFacePoints
    void getMasks();

    // Creates warpped masks out of masks created in getMasks to switch places
    void getWarppedMasks();

    // Returns Mat of refined mask such that warpped mask isn't bigger than original mask
    void getRefinedMasks();

    // Extracts faces from images based on masks created in getMasks
    void extractFaces();

    // Creates warpped faces out of faces extracted in extractFaces
    void getWarppedFaces();

    // Matches src face color to dst face color and vice versa
    void colorCorrectFaces();

    // Blurs edges of mask
    void featherMask(cv::Mat &refined_masks);

    // Pastes faces on original frame
    void pasteFacesOnFrame();

    // Calculates source image histogram and changes target_image to match source hist
    void specifiyHistogram(const cv::Mat source_image, cv::Mat target_image, cv::Mat mask);

    cv::Rect rect_src, rect_dst;
    cv::Rect big_rect_src, big_rect_dst;

    dlib::shape_predictor pose_model;
    dlib::full_object_detection shapes[2];
    dlib::rectangle dlib_rects[2];
    dlib::cv_image<dlib::bgr_pixel> dlib_frame;
    cv::Point2f affine_transform_keypoints_src[3], affine_transform_keypoints_dst[3];

    cv::Mat refined_src_and_dst_warpped, refined_dst_and_src_warpped;
    cv::Mat warpped_face_src, warpped_face_dst;
    
    cv::Point2i points_src[9], points_dst[9];
    cv::Mat trans_src_to_dst, trans_dst_to_src;
    cv::Mat mask_src, mask_dst;
    cv::Mat warpped_mask_src, warpped_mask_dst;

	cv::Mat refined_masks_src;
	cv::Mat refined_masks_dst;


    cv::Mat face_src, face_dst;
    cv::Mat warpped_faces_src, warpped_faces_dst;

    cv::Mat frame_src;
	cv::Mat frame_dst;

	cv::Size frame_size_src;
	cv::Size frame_size_dst;

    cv::Size feather_amount;

    uint8_t LUT[3][256];
    int source_hist_int[3][256];
    int target_hist_int[3][256];
    float source_histogram[3][256];
    float target_histogram[3][256];


	mINI::INIStructure& m_IniFile;

};

