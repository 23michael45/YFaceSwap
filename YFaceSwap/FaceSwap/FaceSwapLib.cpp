#include "FaceSwapLib.h"


bool FaceSwapLib::Init(std::string detectModelPath,std::string alignModelPath)
{
	if(detectModelPath == "")
	{
		m_spDetector = std::make_shared<FaceDetector>("haarcascade_frontalface_default.xml");

	}
	else
	{
		m_spDetector = std::make_shared<FaceDetector>(detectModelPath);
	}

	if(alignModelPath == "")
	{
		m_spFaceExchanger = std::make_shared<FaceExchanger>("shape_predictor_68_face_landmarks.dat");
	}
	else
	{
		m_spFaceExchanger = std::make_shared<FaceExchanger>(alignModelPath);
	}
	
	return true;
}

bool FaceSwapLib::Finalize()
{
	m_spDetector.reset();
	m_spFaceExchanger.reset();

	return true;
}

std::string FaceSwapLib::Calculate(std::string srcPath, std::string dstPath, std::string savePath)
{
	std::string error = "";
	if (m_spFaceExchanger && m_spDetector)
	{
		auto time_start = cv::getTickCount();


		cv::Mat imgSrc = cv::imread(srcPath);
		cv::Mat imgDst = cv::imread(dstPath);

		m_spDetector->detect(imgSrc);

		auto srcfaces = m_spDetector->faces();
		if (srcfaces.size() == 0)
		{
			error = "src file no face\n";
			printf(error.c_str());
			return error;
		}
		else
		{

			printf("src file face num:%i\n", srcfaces.size());
		}

		auto srcFace = srcfaces[0];

		m_spDetector->detect(imgDst);



		auto dstfaces = m_spDetector->faces();
		if (dstfaces.size() == 0)
		{
			error = "dst file no face\n";
			printf(error.c_str());
			return error;
		}
		else
		{

			printf("dst file face num:%i\n", dstfaces.size());
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

			error = "error save";
			printf(error.c_str());
			return error;
		}

	}
	else
	{
		return error;
	}
}
