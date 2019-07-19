#ifndef FaceSwapLib_h__
#define FaceSwapLib_h__
#include <string>
#include "FaceExchanger.h"
#include "FaceDetector.h"
#include "App/ini.h"
class FaceSwapLib
{

public:
	FaceSwapLib();

	bool Init(std::string detectModelPath,std::string alignModelPath,std::string configINIPath);
	bool Finalize();

	bool ReloadINI(std::string configINIPath);

	std::string Calculate(std::string srcPath, std::string dstPath,cv::Mat &result);
	
	std::string Calculate(std::string srcPath, std::string dstPath, std::string savePath);
	std::string CalculateWithMask(std::string srcPath, std::string dstPath, std::string maskPath, std::string savePath);


	mINI::INIStructure m_IniFile;
private:


	std::shared_ptr<FaceDetector> m_spDetector;
	std::shared_ptr<FaceExchanger> m_spFaceExchanger;



};
#endif // FaceSwapLib_h__
