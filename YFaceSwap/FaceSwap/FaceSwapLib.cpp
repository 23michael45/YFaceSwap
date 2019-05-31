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

std::string FaceSwapLib::Calculate(std::string srcPath, std::string dstPath,cv::Mat& result)
{
	try
	{

		std::string error = "";
		if (m_spFaceExchanger && m_spDetector)
		{
			auto time_start = cv::getTickCount();


			cv::Mat imgSrc = cv::imread(srcPath);
			cv::Mat imgDst = cv::imread(dstPath);

			if (imgSrc.cols == 0 || imgSrc.rows == 0 || imgDst.cols == 0 || imgDst.rows == 0)
			{
				return "file not exist";
			}


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

			result = imgDst.clone();
			return "";
		}
	}
	catch (std::exception* e)
	{
		return "error";
	}
}

std::string FaceSwapLib::Calculate(std::string srcPath, std::string dstPath, std::string savePath)
{
	try
	{
		std::string error;

		cv::Mat result;
		error = Calculate(srcPath, dstPath, result);

		if (error.empty())
		{
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
	catch (std::exception* e)
	{
		return "unknown error";
	}
}

std::string FaceSwapLib::CalculateWithMask(std::string srcPath, std::string dstPath, std::string maskPath, std::string savePath)
{
	std::string error;
	try
	{
		if (srcPath == dstPath ||
			dstPath == maskPath ||
			srcPath == maskPath ||
			savePath == srcPath ||
			savePath == dstPath ||
			savePath == maskPath )
		{
			error = "input files path can not be same";
			printf(error.c_str());
			return error;
		}


		cv::Mat result;
		error = Calculate(srcPath, dstPath, result);

		if (error.empty())
		{
			if (maskPath.find(".png") == std::string::npos)
			{
				error = "mask not contain alpha,need png";
				printf(error.c_str());
				return error;

			}

			printf("\nread mask");
			cv::Mat imgMask = cv::imread(maskPath, CV_LOAD_IMAGE_UNCHANGED);
			cv::Mat bgra[4];   //destination array
			split(imgMask, bgra);//split source

			if (imgMask.cols == 0 || imgMask.rows == 0)
			{
				error =  "mask not exist";
				printf(error.c_str());
				return error;
			}

			if (imgMask.cols != result.cols || imgMask.rows != result.rows)
			{
				error = "mask and target not same size";
				printf(error.c_str());
				return error;
			}

			//std::cout << bgra[0];

			printf("\nmask alpha1");
			auto maskA1 = (255 - bgra[3]) / 255;
			cv::Mat maskA3;
			cv::cvtColor(maskA1, maskA3, cv::COLOR_GRAY2BGR);
			cv::multiply(result, maskA3, result);


			printf("\nmask alpha2");
			auto maskB1 = (bgra[3]) / 255;
			cv::Mat maskB3;
			cv::cvtColor(maskB1, maskB3, cv::COLOR_GRAY2BGR);


			printf("\nmask blend");
			cv::Mat imgMask3;
			cv::cvtColor(imgMask, imgMask3, cv::COLOR_BGRA2BGR);
			cv::multiply(imgMask3, maskB3, imgMask);


			printf("\nmask add");
			cv::add(result, imgMask, result);


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
			printf(error.c_str());
			return error;
		}



	}
	catch (std::exception* e)
	{

		error = "unknown error";
		printf(error.c_str());
		return error;
	}
}
