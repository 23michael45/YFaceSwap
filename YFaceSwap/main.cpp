#include <opencv2/highgui/highgui.hpp>

#include "FaceDetectorAndTracker.h"
#include "FaceSwapper.h"

#include "FaceDetector.h".
#include "FaceExchanger.h"

#include <dirent.h>
#include <sys/types.h>
#include "FaceSwapLib.h"
#include "App/ini.h"

//#include <filesystem>
using namespace std; 
//namespace fs = std::filesystem;

FaceSwapLib gFacSwapLib;

int getdir(std::string dir,std::vector<std::string> &files)
{
	DIR *dp;
	struct dirent *dirp;
	if((dp = opendir(dir.c_str())) == NULL)
	{
		cout << "Error";
	}
	while((dirp = readdir(dp)) != NULL)
	{
		files.push_back(string(dirp->d_name));
	}
	closedir(dp);
	return 0;

}


int swap()
{

	try
	{

		FaceDetector detector("haarcascade_frontalface_default.xml");
		FaceSwapper face_swapper("shape_predictor_68_face_landmarks.dat");


		auto time_start = cv::getTickCount();

		// Grab a frame
		cv::Mat frame = cv::imread("images/before.jpg");
		detector.detect(frame);

		auto cv_faces = detector.faces();


		//auto facerect = cv_faces[0];
		//cv::rectangle(frame, facerect.tl(), facerect.br(), (55, 255, 155), 5);
		//cv::imshow("r", frame);
		//cv::waitKey();

		if (cv_faces.size() == 2)
		{
			face_swapper.swapFaces(frame, cv_faces[0], cv_faces[1]);
		}


		// Display it all on the screen
		cv::imshow("Face Swap", frame); 
		cv::waitKey();

		if (cv::waitKey(1) == 27) return 0;

	}
    catch (exception& e)
    {
        cout << e.what() << endl;
    }
}


vector<string> split(const string &s, const string &seperator) {
	vector<string> result;
	typedef string::size_type string_size;
	string_size i = 0;

	while (i != s.size()) {
		//�ҵ��ַ������׸������ڷָ�������ĸ��
		int flag = 0;
		while (i != s.size() && flag == 0) {
			flag = 1;
			for (string_size x = 0; x < seperator.size(); ++x)
				if (s[i] == seperator[x]) {
					++i;
					flag = 0;
					break;
				}
		}

		//�ҵ���һ���ָ������������ָ���֮����ַ���ȡ����
		flag = 0;
		string_size j = i;
		while (j != s.size() && flag == 0) {
			for (string_size x = 0; x < seperator.size(); ++x)
				if (s[j] == seperator[x]) {
					flag = 1;
					break;
				}
			if (flag == 0)
				++j;
		}
		if (i != j) {
			result.push_back(s.substr(i, j - i));
			i = j;
		}
	}
	return result;
}


int main()
{
	SetConsoleOutputCP(CP_UTF8);

	std::ifstream ifs;
	ifs.open("AppConfig.ini", std::ifstream::in);
	if (ifs.is_open())
	{

	}
	else
	{
		mINI::INIFile fileSave("AppConfig.ini");
		mINI::INIStructure iniSave;
		iniSave["images"]["source"] = std::string("images/test01/user1.jpg");
		iniSave["images"]["dest"] = std::string("images/test01/bk.jpg");
		iniSave["images"]["out"] = std::string("images/test01/out.jpg");


		iniSave["points"]["1"] = std::string("9");
		iniSave["points"]["2"] = std::string("37");
		iniSave["points"]["3"] = std::string("46");

		iniSave["offsets"]["v1"] = std::string("0.35");
		iniSave["offsets"]["h1"] = std::string("0");
		iniSave["offsets"]["v2"] = std::string("0");
		iniSave["offsets"]["h2"] = std::string("0.035");
		iniSave["offsets"]["v3"] = std::string("0");
		iniSave["offsets"]["h3"] = std::string("-0.035");


		iniSave["feather"]["scale"] = std::string("0.095");
		fileSave.write(iniSave);

	}
	ifs.close();


	gFacSwapLib.Init("haarcascade_frontalface_default.xml", "shape_predictor_68_face_landmarks.dat","AppConfig.ini");

	do 
	{


		gFacSwapLib.ReloadINI("AppConfig.ini");
		std::string srcImage = gFacSwapLib.m_IniFile["images"]["source"];
		std::string dstImage = gFacSwapLib.m_IniFile["images"]["dest"];
		std::string outImage = gFacSwapLib.m_IniFile["images"]["out"];
		std::string maskImage = gFacSwapLib.m_IniFile["images"]["mask"];


		string retpath = "";
		if (maskImage == "")
		{

			retpath = gFacSwapLib.Calculate(srcImage, dstImage, outImage);
		}
		else
		{

			retpath = gFacSwapLib.CalculateWithMask(srcImage, dstImage, maskImage,outImage);
		}
		auto retImg = cv::imread(retpath);

		cv::imshow("ret", retImg);

		printf("修改AppConfig.ini后，按回车重新计算！");

		cv::waitKey();

		cvDestroyWindow("ret");
	} while (true);
	
	

}

int main_dep()
{
	/*FaceDetector detector(haarcascade_frontalface_default.xml);
	FaceExchanger exchanger("shape_predictor_68_face_landmarks.dat");*/

	gFacSwapLib.Init("haarcascade_frontalface_default.xml","shape_predictor_68_face_landmarks.dat","AppConfig.ini");

	int key = 0;
	do 
	{
		printf("input: 1 exchange 2 swap");
		key = getchar();
		if (key == 0x31)
		{
			printf("exchanger : input 4 file name   a:b:c:d\n");

			std::this_thread::sleep_for(std::chrono::milliseconds(500));


			std::string cmd;
			std::cin >> cmd;

			auto arr = split(cmd, ":");


			//images/src/1.jpg:images/dst/1.png
			if (arr.size() == 3)
			{
				std::string srcfile = arr[0];
				std::string dstfile = arr[1];
				std::string savefile = arr[2];


				string retpath = gFacSwapLib.Calculate(srcfile, dstfile, savefile);

				if (retpath != savefile)
				{
					std::cout << "error ret" << std::endl;
				}
				else
				{
					auto retImg = cv::imread(retpath);
					cv::imshow("ret", retImg);
					cv::waitKey();

				}
			}
			if (arr.size() == 4)
			{
				std::string srcfile = arr[0];
				std::string dstfile = arr[1];
				std::string maskfile = arr[2];
				std::string savefile = arr[3];


				string retpath = gFacSwapLib.CalculateWithMask(srcfile, dstfile,maskfile, savefile);

				if (retpath != savefile)
				{
					std::cout << "error:" << retpath << std::endl;
				}
				else
				{
					auto retImg = cv::imread(retpath);
					cv::imshow("ret", retImg);
					cv::waitKey();

				}
			}
			//else if (arr.size() ==  4)
			//{
			//	std::string srcfile = arr[0];
			//	std::string dstpath = arr[1];
			//	std::string maskpath = arr[2];
			//	std::string savefile = arr[3];

			//	// if (!fs::exists(genpath))
			//	// {
			//	// 	fs::create_directory(genpath);

			//	// }
			//	std::vector<std::string> files;
			//	getdir(dstpath,files);
			//	int i = 0;
			//	for (const auto & entry : files)//fs::directory_iterator(dstpath))
			//	{
			//		// std::cout << entry.path().string() << std::endl;
			//		std::cout << entry << std::endl;

			//		cv::Mat ret;
			//		// exchange(detector, exchanger, srcfile, entry.path().string(),ret);
			//		exchange(detector, exchanger, srcfile, entry,ret);

			//		i++;


			//		char buffer[256];
			//		sprintf(buffer, "%s/%i.jpg",genpath.c_str(),i);
			//		string savefile = buffer;
			//		imwrite(savefile, ret);
			//	}


			//}
		}
		else if (key == 0x32)
		{
			printf("swapper");
			swap();
		}


		std::this_thread::sleep_for(std::chrono::milliseconds(500));
	} while (key != 0x39);

	gFacSwapLib.Finalize();

	return 0;
}