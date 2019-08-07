#include "FaceExchanger.h"

#include <iostream>
#include "App/ini.h"


#define ShowAndClose(name,mat) 	cv::imshow(##name, mat);\
cv::waitKey();\
cv::destroyWindow(##name);

FaceExchanger::FaceExchanger(const std::string landmarks_path, mINI::INIStructure& iniFile):m_IniFile(iniFile)
{

    try
    {
        dlib::deserialize(landmarks_path) >> pose_model;
    }
    catch (std::exception& e)
    {
        std::cerr << "Error loading landmarks from " << landmarks_path << std::endl
            << "You can download the file from http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2" << std::endl;
        exit(-1);
    }
}


FaceExchanger::~FaceExchanger()
{
}

void FaceExchanger::swapFaces(cv::Mat &src, cv::Mat &dst, cv::Rect &rect_src, cv::Rect &rect_dst)
{
	frame_src = src;
	frame_dst = dst;
    //small_frame = getMinFrame(frame, rect_src, rect_dst);
	this->rect_src = rect_src;
	this->rect_dst = rect_dst;
	frame_size_src = cv::Size(src.cols, src.rows);
    frame_size_dst = cv::Size(dst.cols, dst.rows);


	big_rect_src = ((this->rect_src - cv::Point(rect_src.width / 4, rect_src.height / 4)) + cv::Size(rect_src.width / 2, rect_src.height / 2)) & cv::Rect(0, 0, frame_size_src.width, frame_size_src.height);
	big_rect_dst = ((this->rect_dst - cv::Point(rect_dst.width / 4, rect_dst.height / 4)) + cv::Size(rect_dst.width / 2, rect_dst.height / 2)) & cv::Rect(0, 0, frame_size_dst.width, frame_size_dst.height);


	//cv::rectangle(src, big_rect_src.tl(), big_rect_src.br(), (55, 255, 155), 5);
	//cv::imshow("src", src);

	//cv::rectangle(dst, big_rect_dst.tl(), big_rect_dst.br(), (55, 255, 155), 5);
	//cv::imshow("imgDst", dst);
	//cv::waitKey();

    getFacePoints(src,dst);

    getTransformationMatrices();

    mask_src.create(frame_size_src, CV_8UC1);
    mask_dst.create(frame_size_dst, CV_8UC1);
    getMasks();

    getWarppedMasks();

    getRefinedMasks();

    extractFaces();

    getWarppedFaces();

    colorCorrectFaces();

    auto refined_mask_src_big = refined_masks_src(big_rect_src);
    auto refined_mask_dst_big = refined_masks_dst(big_rect_dst);


	//cv::imshow("refined_mask_src_big", refined_mask_src_big);
	//cv::imshow("refined_mask_dst_big", refined_mask_dst_big);


    featherMask(refined_mask_src_big);
    featherMask(refined_mask_dst_big);

    pasteFacesOnFrame();
}

cv::Mat FaceExchanger::getMinFrame(const cv::Mat &frame, cv::Rect &rect_src, cv::Rect &rect_dst)
{
    cv::Rect bounding_rect = rect_src | rect_dst;

    bounding_rect -= cv::Point(50, 50);
    bounding_rect += cv::Size(100, 100);

    bounding_rect &= cv::Rect(0, 0, frame.cols, frame.rows);

    this->rect_src = rect_src - bounding_rect.tl();
    this->rect_dst = rect_dst - bounding_rect.tl();

    big_rect_src = ((this->rect_src - cv::Point(rect_src.width / 4, rect_src.height / 4)) + cv::Size(rect_src.width / 2, rect_src.height / 2)) & cv::Rect(0, 0, bounding_rect.width, bounding_rect.height);
    big_rect_dst = ((this->rect_dst - cv::Point(rect_dst.width / 4, rect_dst.height / 4)) + cv::Size(rect_dst.width / 2, rect_dst.height / 2)) & cv::Rect(0, 0, bounding_rect.width, bounding_rect.height);



    return frame(bounding_rect);
}


std::vector<std::string> split(const std::string &text, char sep)
{
	std::vector<std::string> tokens;
	std::size_t start = 0, end = 0;
	while ((end = text.find(sep, start)) != std::string::npos) {
		if (end != start) {
			tokens.push_back(text.substr(start, end - start));
		}
		start = end + 1;
	}
	if (end != start) {
		tokens.push_back(text.substr(start));
	}
	return tokens;
}


void FaceExchanger::getFacePoints(const cv::Mat &src,const cv::Mat &dst)
{
    using namespace dlib;

    dlib_rects[0] = rectangle(rect_src.x, rect_src.y, rect_src.x + rect_src.width, rect_src.y + rect_src.height);
    dlib_rects[1] = rectangle(rect_dst.x, rect_dst.y, rect_dst.x + rect_dst.width, rect_dst.y + rect_dst.height);


	dlib::cv_image<dlib::bgr_pixel> dlib_src = src;
	dlib::cv_image<dlib::bgr_pixel> dlib_dst = dst;

    shapes[0] = pose_model(dlib_src, dlib_rects[0]);
    shapes[1] = pose_model(dlib_dst, dlib_rects[1]);




    auto getPoint = [&](int shape_index, int part_index) -> const cv::Point2i
    {
        const auto &p = shapes[shape_index].part(part_index);
        return cv::Point2i(p.x(), p.y());
    };


	//cv::Mat srcdraw;
	//src.copyTo(srcdraw);
	//cv::rectangle(srcdraw, rect_src.tl(), rect_src.br(), (55, 255, 155), 5);
	//for (int i = 0; i < 68; i++)
	//{
	//	cv::circle(srcdraw, getPoint(0, i), 2, cv::Scalar(0, 255, 0));
	//}
	//ShowAndClose("face keypoint src", srcdraw)



	//	cv::Mat dstdraw;
	//dst.copyTo(dstdraw);
	//cv::rectangle(dstdraw, rect_dst.tl(), rect_dst.br(), (55, 255, 155), 5);
	//for (int i = 0; i < 68; i++)
	//{
	//	cv::circle(dstdraw, getPoint(1, i), 2, cv::Scalar(0, 255, 0));
	//}
	//ShowAndClose("face keypoint dst", dstdraw)




	float g_feater_scale = atof(m_IniFile["feather"]["scale"].c_str());

    points_src[0] = getPoint(0, FACE_GET(LEFT_FACE_CONTOUR_1));
    points_src[1] = getPoint(0, FACE_GET(LEFT_FACE_CONTOUR_4));
    points_src[2] = getPoint(0, FACE_GET(LEFT_FACE_CONTOUR_6));
    points_src[3] = getPoint(0, FACE_GET(MIDDLE_CHIN));
    points_src[4] = getPoint(0, FACE_GET(RIGHT_FACE_CONTOUR_6));
    points_src[5] = getPoint(0, FACE_GET(RIGHT_FACE_CONTOUR_4));
    points_src[6] = getPoint(0, FACE_GET(RIGHT_FACE_CONTOUR_1));

    cv::Point2i src_nose_length = getPoint(0, FACE_GET(MIDDLE_NOSE_1)) - getPoint(0, FACE_GET(MIDDLE_NOSE_4));

	cv::Point2i src_lefteye_length = getPoint(0, FACE_GET(LEFT_EYE_4)) - getPoint(0, FACE_GET(LEFT_EYE_1));
	cv::Point2i src_righteye_length = getPoint(0, FACE_GET(RIGHT_EYE_4)) - getPoint(0, FACE_GET(RIGHT_EYE_1));

	cv::Point2i src_eye_length = getPoint(0, FACE_GET(RIGHT_EYE_1)) - getPoint(0, FACE_GET(LEFT_EYE_1));
	cv::Point2i src_chin_length = getPoint(0, FACE_GET(OUT_DOWN_MOUTH)) - getPoint(0, FACE_GET(MIDDLE_CHIN));
    points_src[7] = getPoint(0, FACE_GET(RIGHT_EYEBROW_1)) + src_nose_length;
    points_src[8] = getPoint(0, FACE_GET(LEFT_EYEBROW_1)) + src_nose_length;


	for (int i = 1; i <= 3; i++)
	{
		std::stringstream ss;
		ss << i;
		std::string str = ss.str();


		std::string s = m_IniFile["points"][str];
		auto vec = split(s, ' ');


		int count = 0;
		cv::Point2i pt(0,0);
		for (auto index : vec)
		{
			int keypointIdx = atoi(index.c_str());

			pt += getPoint(0, FACE_GET(keypointIdx));
			count++;
		}
		pt = pt / count;


		affine_transform_keypoints_src[i-1] = pt;

	}



	//---------------------------------------------------------------------------------------------------
	points_dst[0] = getPoint(1, FACE_GET(LEFT_FACE_CONTOUR_1));
	points_dst[1] = getPoint(1, FACE_GET(LEFT_FACE_CONTOUR_4));
	points_dst[2] = getPoint(1, FACE_GET(LEFT_FACE_CONTOUR_6));
	points_dst[3] = getPoint(1, FACE_GET(MIDDLE_CHIN));
	points_dst[4] = getPoint(1, FACE_GET(RIGHT_FACE_CONTOUR_6));
	points_dst[5] = getPoint(1, FACE_GET(RIGHT_FACE_CONTOUR_4));
	points_dst[6] = getPoint(1, FACE_GET(RIGHT_FACE_CONTOUR_1));

	cv::Point2i dst_nose_length = getPoint(1, FACE_GET(MIDDLE_NOSE_1)) - getPoint(1, FACE_GET(MIDDLE_NOSE_4));
	cv::Point2i dst_lefteye_length = getPoint(1, FACE_GET(LEFT_EYE_4)) - getPoint(1, FACE_GET(LEFT_EYE_1));
	cv::Point2i dst_righteye_length = getPoint(1, FACE_GET(RIGHT_EYE_4)) - getPoint(1, FACE_GET(RIGHT_EYE_1));


	cv::Point2i dst_eye_length = getPoint(1, FACE_GET(RIGHT_EYE_1)) - getPoint(1, FACE_GET(LEFT_EYE_1));
	cv::Point2i dst_chin_length = getPoint(1, FACE_GET(OUT_DOWN_MOUTH)) - getPoint(1, FACE_GET(MIDDLE_CHIN));

	points_dst[7] = getPoint(1, FACE_GET(RIGHT_EYEBROW_1)) + dst_nose_length;
	points_dst[8] = getPoint(1, FACE_GET(LEFT_EYEBROW_1)) + dst_nose_length;



	for (int i = 1; i <= 3; i++)
	{
		std::stringstream ss;
		ss << i;
		std::string str = ss.str();


		std::string s = m_IniFile["points"][str];
		auto vec = split(s, ' ');


		int count = 0;
		cv::Point2i pt(0, 0);
		for (auto index : vec)
		{
			int keypointIdx = atoi(index.c_str());

			pt += getPoint(1, FACE_GET(keypointIdx));
			count++;
		}
		pt = pt / count;




		float g_offset_v = atof(m_IniFile["offsets"]["v"+str].c_str());
		float g_offset_h = atof(m_IniFile["offsets"]["h" + str].c_str());

		affine_transform_keypoints_dst[i-1] = pt + dst_chin_length * g_offset_v + dst_eye_length * g_offset_h;

	}


	feather_amount.width = feather_amount.height = (int)cv::norm(points_dst[0] - points_dst[6])  * g_feater_scale;
}

void FaceExchanger::getTransformationMatrices()
{
    trans_src_to_dst = cv::getAffineTransform(affine_transform_keypoints_src, affine_transform_keypoints_dst);
    cv::invertAffineTransform(trans_src_to_dst, trans_dst_to_src);
}

void FaceExchanger::getMasks()
{
    mask_src.setTo(cv::Scalar::all(0));
    mask_dst.setTo(cv::Scalar::all(0));

    cv::fillConvexPoly(mask_src, points_src, 9, cv::Scalar(255));
    cv::fillConvexPoly(mask_dst, points_dst, 9, cv::Scalar(255));

	//ShowAndClose("mask_src", mask_src)
	//ShowAndClose("mask_dst", mask_dst);
}

void FaceExchanger::getWarppedMasks()
{

    cv::warpAffine(mask_src, warpped_mask_src, trans_src_to_dst, frame_size_dst, cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));
    cv::warpAffine(mask_dst, warpped_mask_dst, trans_dst_to_src, frame_size_src, cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));

	//ShowAndClose("warpped_mask_src", warpped_mask_src)
	//ShowAndClose("warpped_mask_dst", warpped_mask_dst)
}

void FaceExchanger::getRefinedMasks()
{
    cv::bitwise_and(mask_src, warpped_mask_dst, refined_src_and_dst_warpped);
    cv::bitwise_and(mask_dst, warpped_mask_src, refined_dst_and_src_warpped);



	cv::Mat refined_masks_src_temp(frame_size_src, CV_8UC1, cv::Scalar(0));
    cv::Mat refined_masks_dst_temp(frame_size_dst, CV_8UC1, cv::Scalar(0));


    refined_src_and_dst_warpped.copyTo(refined_masks_src_temp, refined_src_and_dst_warpped);
    refined_dst_and_src_warpped.copyTo(refined_masks_dst_temp, refined_dst_and_src_warpped);

	refined_masks_src = refined_masks_src_temp;
	refined_masks_dst = refined_masks_dst_temp;


	//ShowAndClose("refined_masks_src", refined_masks_src)
	//ShowAndClose("refined_masks_dst", refined_masks_dst)
}

void FaceExchanger::extractFaces()
{
    frame_src.copyTo(face_src, mask_src);
    frame_dst.copyTo(face_dst, mask_dst);


	//ShowAndClose("extractFaces src", face_src)
	//	ShowAndClose("extractFaces dst", face_dst)
}

void FaceExchanger::getWarppedFaces()
{
	cv::Mat warpped_faces_src_temp(frame_size_src, CV_8UC3, cv::Scalar::all(0));
	cv::Mat warpped_faces_dst_temp(frame_size_dst, CV_8UC3, cv::Scalar::all(0));

    cv::warpAffine(face_src, warpped_face_src, trans_src_to_dst, frame_size_dst, cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    cv::warpAffine(face_dst, warpped_face_dst, trans_dst_to_src, frame_size_src, cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    warpped_face_src.copyTo(warpped_faces_src_temp, warpped_mask_src);
    warpped_face_dst.copyTo(warpped_faces_dst_temp, warpped_mask_dst);


    warpped_faces_src = warpped_faces_src_temp;
	warpped_faces_dst = warpped_faces_dst_temp;

	//ShowAndClose("warpped_faces_src", warpped_faces_src)
	//ShowAndClose("warpped_faces_dst", warpped_faces_dst)
}

void FaceExchanger::colorCorrectFaces()
{


	//ShowAndClose("frame_src(big_rect_src)", frame_src(big_rect_src))
	//ShowAndClose("warpped_faces_dst(big_rect_src)", warpped_faces_dst(big_rect_src))
	//ShowAndClose("warpped_mask_dst(big_rect_src)", warpped_mask_dst(big_rect_src))

	specifiyHistogram(frame_src(big_rect_src), warpped_faces_dst(big_rect_src), warpped_mask_dst(big_rect_src));



	//ShowAndClose("frame_dst(big_rect_src)", frame_dst(big_rect_dst))
	//	ShowAndClose("warpped_faces_src(big_rect_dst)", warpped_faces_src(big_rect_dst))
	//	ShowAndClose("warpped_mask_src(big_rect_dst)", warpped_mask_src(big_rect_dst))


    specifiyHistogram(frame_dst(big_rect_dst), warpped_faces_src(big_rect_dst), warpped_mask_src(big_rect_dst));

}

void FaceExchanger::featherMask(cv::Mat &refined_masks)
{
    cv::erode(refined_masks, refined_masks, getStructuringElement(cv::MORPH_RECT, feather_amount), cv::Point(-1, -1), 1, cv::BORDER_CONSTANT, cv::Scalar(0));

    cv::blur(refined_masks, refined_masks, feather_amount, cv::Point(-1, -1), cv::BORDER_CONSTANT);


	//ShowAndClose("featherMask", refined_masks)
}

inline void FaceExchanger::pasteFacesOnFrame()
{

	//ShowAndClose("frame_dst", frame_dst)
	//	ShowAndClose("warpped_faces_src", warpped_faces_src)
	//	ShowAndClose("refined_masks_dst", refined_masks_dst)

    for (size_t i = 0; i < frame_dst.rows; i++)
    {
        auto frame_pixel = frame_dst.row(i).data;
        auto faces_pixel = warpped_faces_src.row(i).data;
        auto masks_pixel = refined_masks_dst.row(i).data;

        for (size_t j = 0; j < frame_dst.cols; j++)
        {
            if (*masks_pixel != 0)
			{


                *frame_pixel = ((255 - *masks_pixel) * (*frame_pixel) + (*masks_pixel) * (*faces_pixel)) >> 8; // divide by 256
                *(frame_pixel + 1) = ((255 - *(masks_pixel + 1)) * (*(frame_pixel + 1)) + (*(masks_pixel + 1)) * (*(faces_pixel + 1))) >> 8;
                *(frame_pixel + 2) = ((255 - *(masks_pixel + 2)) * (*(frame_pixel + 2)) + (*(masks_pixel + 2)) * (*(faces_pixel + 2))) >> 8;

            }

            frame_pixel += 3;
            faces_pixel += 3;
            masks_pixel++;
        }
    }
}

void FaceExchanger::specifiyHistogram(const cv::Mat source_image, cv::Mat target_image, cv::Mat mask)
{

	//ShowAndClose("source_image",source_image)
	//ShowAndClose("target_image",target_image)
	//ShowAndClose("mask",mask)

    std::memset(source_hist_int, 0, sizeof(int) * 3 * 256);
    std::memset(target_hist_int, 0, sizeof(int) * 3 * 256);

    for (size_t i = 0; i < mask.rows; i++)
    {
        auto current_mask_pixel = mask.row(i).data;
        auto current_source_pixel = source_image.row(i).data;
        auto current_target_pixel = target_image.row(i).data;

        for (size_t j = 0; j < mask.cols; j++)
        {
            if (*current_mask_pixel != 0) {
                source_hist_int[0][*current_source_pixel]++;
                source_hist_int[1][*(current_source_pixel + 1)]++;
                source_hist_int[2][*(current_source_pixel + 2)]++;

                target_hist_int[0][*current_target_pixel]++;
                target_hist_int[1][*(current_target_pixel + 1)]++;
                target_hist_int[2][*(current_target_pixel + 2)]++;
            }

            // Advance to next pixel
            current_source_pixel += 3; 
            current_target_pixel += 3; 
            current_mask_pixel++; 
        }
    }

    // Calc CDF  (Cumulative Distribution Function)
    for (size_t i = 1; i < 256; i++)
    {
        source_hist_int[0][i] += source_hist_int[0][i - 1];
        source_hist_int[1][i] += source_hist_int[1][i - 1];
        source_hist_int[2][i] += source_hist_int[2][i - 1];

        target_hist_int[0][i] += target_hist_int[0][i - 1];
        target_hist_int[1][i] += target_hist_int[1][i - 1];
        target_hist_int[2][i] += target_hist_int[2][i - 1];
    }

    // Normalize CDF
    for (size_t i = 0; i < 256; i++)
    {
        source_histogram[0][i] = (source_hist_int[0][255] ? (float)source_hist_int[0][i] / source_hist_int[0][255] : 0);
        source_histogram[1][i] = (source_hist_int[1][255] ? (float)source_hist_int[1][i] / source_hist_int[1][255] : 0);
        source_histogram[2][i] = (source_hist_int[2][255] ? (float)source_hist_int[2][i] / source_hist_int[2][255] : 0);

        target_histogram[0][i] = (target_hist_int[0][255] ? (float)target_hist_int[0][i] / target_hist_int[0][255] : 0);
        target_histogram[1][i] = (target_hist_int[1][255] ? (float)target_hist_int[1][i] / target_hist_int[1][255] : 0);
        target_histogram[2][i] = (target_hist_int[2][255] ? (float)target_hist_int[2][i] / target_hist_int[2][255] : 0);
    }

    // Create lookup table

    auto binary_search = [&](const float needle, const float haystack[]) -> uint8_t
    {
        uint8_t l = 0, r = 255, m;
        while (l < r)
        {
            m = (l + r) / 2;
            if (needle > haystack[m])
                l = m + 1;
            else
                r = m - 1;
        }
        // TODO check closest value
        return m;
    };

    for (size_t i = 0; i < 256; i++)
    {
        LUT[0][i] = binary_search(target_histogram[0][i], source_histogram[0]);
        LUT[1][i] = binary_search(target_histogram[1][i], source_histogram[1]);
        LUT[2][i] = binary_search(target_histogram[2][i], source_histogram[2]);
    }

    // repaint pixels
    for (size_t i = 0; i < mask.rows; i++)
    {
        auto current_mask_pixel = mask.row(i).data;
        auto current_target_pixel = target_image.row(i).data;
        for (size_t j = 0; j < mask.cols; j++)
        {
            if (*current_mask_pixel != 0)
            {
                *current_target_pixel = LUT[0][*current_target_pixel];
                *(current_target_pixel + 1) = LUT[1][*(current_target_pixel + 1)];
                *(current_target_pixel + 2) = LUT[2][*(current_target_pixel + 2)];
            }

            // Advance to next pixel
            current_target_pixel += 3;
            current_mask_pixel++;
        }
    }
}
