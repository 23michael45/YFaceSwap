#ifndef FaceSwapLib_h__
#define FaceSwapLib_h__
#include <string>
#include "FaceExchanger.h"
#include "FaceDetector.h"
class FaceSwapLib
{

public:
	bool Init();
	bool Finalize();

	std::string Calculate(std::string srcPath, std::string dstPath, std::string savePath);


private:


	std::shared_ptr<FaceDetector> m_spDetector;
	std::shared_ptr<FaceExchanger> m_spFaceExchanger;
};
#endif // FaceSwapLib_h__
