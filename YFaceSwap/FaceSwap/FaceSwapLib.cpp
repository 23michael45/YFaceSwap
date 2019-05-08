#include "FaceSwapLib.h"


bool FaceSwapLib::Init()
{
	m_spDetector = std::make_shared<FaceDetector>("haarcascade_frontalface_default.xml");
	m_spFaceExchanger = std::make_shared<FaceExchanger>("shape_predictor_68_face_landmarks.dat");

	return true;
}

bool FaceSwapLib::Finalize()
{
	m_spDetector.reset();
	m_spFaceExchanger.reset();
}

std::string FaceSwapLib::Calculate(std::string srcPath, std::string dstPath, std::string savePath)
{
	if (m_spFaceExchanger && m_spDetector)
	{
		auto time_start = cv::getTickCount();


		cv::Mat imgSrc = cv::imread(srcPath);
		cv::Mat imgDst = cv::imread(dstPath);

		m_spDetector->detect(imgSrc);

		auto srcfaces = m_spDetector->faces();
		if (srcfaces.size() == 0)
		{
			printf("src file no face");
			return 0;
		}
		else
		{

			printf("src file face num:%i", srcfaces.size());
		}

		auto srcFace = srcfaces[0];

		m_spDetector->detect(imgDst);



		auto dstfaces = m_spDetector->faces();
		if (dstfaces.size() == 0)
		{
			printf("dst file no face");
			return 0;
		}
		else
		{

			printf("dst file face num:%i", dstfaces.size());
		}

		auto dstFace = dstfaces[0];

		m_spFaceExchanger->swapFaces(imgSrc, imgDst, srcFace, dstFace);

		cv::Mat result;
		imgDst.copyTo(result);

		bool b = cv::imwrite(savePath, result);
		if (b)
		{
			return savePath;
		}
		else
		{
			return "error save";
		}

	}
	else
	{
		return "";
	}
}
