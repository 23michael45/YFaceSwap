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
//#include <dlib/gui_widgets.h>

class FaceExchanger
{
public:
    // Initialize face swapped with landmarks
    FaceExchanger(const std::string landmarks_path);
    ~FaceExchanger();

    //Swaps faces in rects on frame
    void swapFaces(cv::Mat &src,cv::Mat &dst, cv::Rect &rect_src, cv::Rect &rect_dst);

private:
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
};

