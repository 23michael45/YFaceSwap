#include "FaceSwapLib.h"


FaceSwapLib::FaceSwapLib()
{

}

bool FaceSwapLib::Init(std::string detectModelPath,std::string alignModelPath, std::string configINIPath)
{
	if (configINIPath == "")
	{
		m_IniFile["images"]["source"] = std::string("images/test01/user1.jpg");
		m_IniFile["images"]["dest"] = std::string("images/test01/bk.jpg");
		m_IniFile["images"]["out"] = std::string("images/test01/out.jpg");


		m_IniFile["points"]["1"] = std::string("43 46");
		m_IniFile["points"]["2"] = std::string("40 37");
		m_IniFile["points"]["3"] = std::string("63 67");

		m_IniFile["offsets"]["v1"] = std::string("*0.05");
		m_IniFile["offsets"]["h1"] = std::string("0.015");
		m_IniFile["offsets"]["v2"] = std::string("-0.05");
		m_IniFile["offsets"]["h2"] = std::string("0.015");
		m_IniFile["offsets"]["v3"] = std::string("0");
		m_IniFile["offsets"]["h3"] = std::string("0");

		m_IniFile["feather"]["scale"] = std::string("0.125");

	}
	else
	{
		ReloadINI(configINIPath);
	}

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
		m_spFaceExchanger = std::make_shared<FaceExchanger>("shape_predictor_68_face_landmarks.dat", m_IniFile);
	}
	else
	{
		m_spFaceExchanger = std::make_shared<FaceExchanger>(alignModelPath, m_IniFile);
	}

	
	return true;
}

bool FaceSwapLib::Finalize()
{
	m_spDetector.reset();
	m_spFaceExchanger.reset();

	return true;
}

bool FaceSwapLib::ReloadINI(std::string configINIPath)
{
	mINI::INIFile file(configINIPath);
	file.read(m_IniFile);
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
			cv::Mat imgMask = cv::imread(maskPath, cv::IMREAD_UNCHANGED);
			cv::Mat imgMask32F;
			imgMask.convertTo(imgMask32F, CV_32FC4);

			cv::Mat alpha;

			if (imgMask.channels() == 4)
			{

				cv::Mat bgra[4];   //destination array
				split(imgMask, bgra);//split source
				alpha = bgra[3];
			}
			else if (imgMask.channels() < 4)
			{
				error = "mask not contain alpha";
				printf(error.c_str());
				return error;
			}


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

			cv::Mat alpha32F;
			alpha.convertTo(alpha32F, CV_32FC1);
			cv::Mat result32F;
			result.convertTo(result32F, CV_32FC3);
			

			printf("\nmask alpha1");
			auto maskA1 = (255.0f - alpha32F) / 255.0f;
			//std::cout << maskA1;
			cv::Mat maskA3;
			cv::cvtColor(maskA1, maskA3, cv::COLOR_GRAY2BGR);
			cv::multiply(result32F, maskA3, result32F);


			printf("\nmask alpha2");
			auto maskB1 = (alpha32F) / 255;
			cv::Mat maskB3;
			cv::cvtColor(maskB1, maskB3, cv::COLOR_GRAY2BGR);


			printf("\nmask blend");
			cv::Mat imgMask3;
			cv::cvtColor(imgMask32F, imgMask3, cv::COLOR_BGRA2BGR);
			cv::multiply(imgMask3, maskB3, imgMask);


			printf("\nmask add");

			cv::add(result32F, imgMask, result32F);
			result32F.convertTo(result, CV_8UC3);


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
