/* DO NOT EDIT THIS FILE - it is machine generated */
#include "com_yfaceswap_FaceSwapLib.h"
#include "FaceSwapLib.h"
/* Header for class com_facecloud_FaceCloudLib */

#ifdef __cplusplus
extern "C" {
#endif


	FaceSwapLib gFacSwapLib;


	std::string jstring2string(JNIEnv *env, jstring jStr) {
		if (!jStr)
			return "";

		const jclass stringClass = env->GetObjectClass(jStr);
		const jmethodID getBytes = env->GetMethodID(stringClass, "getBytes", "(Ljava/lang/String;)[B");
		const jbyteArray stringJbytes = (jbyteArray)env->CallObjectMethod(jStr, getBytes, env->NewStringUTF("UTF-8"));

		size_t length = (size_t)env->GetArrayLength(stringJbytes);
		jbyte* pBytes = env->GetByteArrayElements(stringJbytes, NULL);

		std::string ret = std::string((char *)pBytes, length);
		env->ReleaseByteArrayElements(stringJbytes, pBytes, JNI_ABORT);

		env->DeleteLocalRef(stringJbytes);
		env->DeleteLocalRef(stringClass);
		return ret;
	}




	JNIEXPORT jboolean JNICALL Java_com_yfaceswap_FaceSwapLib_Init(JNIEnv *env, jobject,jstring detectModelPath, jstring alignModelPath)
	{
		return gFacSwapLib.Init(jstring2string(env, detectModelPath),jstring2string(env, alignModelPath));
	}

	JNIEXPORT jboolean JNICALL Java_com_yfaceswap_FaceSwapLib_Finalize(JNIEnv *, jobject)
	{

		return gFacSwapLib.Finalize();
	}

	JNIEXPORT jstring JNICALL Java_com_yfaceswap_FaceSwapLib_Calculate(JNIEnv *env, jobject obj, jstring srcPath, jstring dstPath, jstring savePath)
	{
		std::string ret = gFacSwapLib.Calculate(jstring2string(env, srcPath), jstring2string(env, dstPath), jstring2string(env, savePath));
		return env->NewStringUTF(ret.c_str());
	}




#ifdef __cplusplus
}


#endif
