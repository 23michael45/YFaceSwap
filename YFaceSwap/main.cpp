#include <opencv2/highgui/highgui.hpp>

#include "FaceDetectorAndTracker.h"
#include "FaceSwapper.h"

#include "FaceDetector.h".
#include "FaceExchanger.h"

#include <dirent.h>
#include <sys/types.h>

//#include <filesystem>
using namespace std; 
//namespace fs = std::filesystem;

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

int exchange(FaceDetector& detector,FaceExchanger& exchanger,std::string srcfile,std::string dstfile,cv::Mat& mat)
{
	try
	{

		auto time_start = cv::getTickCount();


		cv::Mat imgSrc = cv::imread(srcfile);
		cv::Mat imgDst = cv::imread(dstfile);

		detector.detect(imgSrc);

		auto srcfaces = detector.faces();
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

		detector.detect(imgDst);



		auto dstfaces = detector.faces();
		if (dstfaces.size() == 0)
		{
			printf("dst file no face");
			return 0;
		}
		else
		{

			printf("dst file face num:%i",dstfaces.size());
		}

		auto dstFace = dstfaces[0];



		//cv::rectangle(imgSrc, srcFace.tl(), srcFace.br(), (55, 255, 155), 5);
		//cv::imshow("imgSrc", imgSrc);
		//cv::waitKey();


		//cv::rectangle(imgDst, dstFace.tl(), dstFace.br(), (55, 255, 155), 5);
		//cv::imshow("imgDst", imgDst);
		//cv::waitKey();



		exchanger.swapFaces(imgSrc, imgDst, srcFace, dstFace);

		//cv::imshow("imgSrc", imgSrc);
		//cv::imshow("imgDst", imgDst);
		//cv::waitKey();

		imgDst.copyTo(mat);
	}
	catch (exception& e)
	{
		cout << e.what() << endl;
	}

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
	FaceDetector detector("haarcascade_frontalface_default.xml");
	FaceExchanger exchanger("shape_predictor_68_face_landmarks.dat");



	int key = 0;
	do 
	{
		printf("input: 1 exchange 2 swap");
		key = getchar();
		if (key == 0x31)
		{
			printf("exchanger : input 2 file name   a:b\n");

			std::this_thread::sleep_for(std::chrono::milliseconds(500));


			std::string cmd;
			std::cin >> cmd;

			auto arr = split(cmd, ":");


			//images/src/1.jpg:images/dst/1.png
			if (arr.size() == 2)
			{
				std::string srcfile = arr[0];
				std::string dstfile = arr[1];

				cv::Mat ret;
				exchange(detector, exchanger, srcfile, dstfile,ret);



				//cv::imshow("imgSrc", imgSrc);
				cv::imshow("ret", ret);
				cv::waitKey();
			}
			else if (arr.size() == 3)
			{
				std::string srcfile = arr[0];
				std::string dstpath = arr[1];
				std::string genpath = arr[2];

				// if (!fs::exists(genpath))
				// {
				// 	fs::create_directory(genpath);

				// }
				std::vector<std::string> files;
				getdir(dstpath,files);
				int i = 0;
				for (const auto & entry : files)//fs::directory_iterator(dstpath))
				{
					// std::cout << entry.path().string() << std::endl;
					std::cout << entry << std::endl;

					cv::Mat ret;
					// exchange(detector, exchanger, srcfile, entry.path().string(),ret);
					exchange(detector, exchanger, srcfile, entry,ret);

					i++;


					char buffer[256];
					sprintf(buffer, "%s/%i.jpg",genpath.c_str(),i);
					string savefile = buffer;
					imwrite(savefile, ret);
				}


			}
		}
		else if (key == 0x32)
		{
			printf("swapper");
			swap();
		}


		std::this_thread::sleep_for(std::chrono::milliseconds(500));
	} while (key != 0x39);
}