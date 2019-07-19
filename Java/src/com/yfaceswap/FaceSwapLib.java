package com.yfaceswap;
public class FaceSwapLib {
	
	
	//初始化换脸计算库
	public native boolean Init(String detectModelPath,String alignModelPath,String iniConfigPath);
	
	//销毁换脸计算库
	public native boolean Finalize();
	
	//计算换脸
	//srcPath 上传的用户相片路径
	//dstPath 要被换脸的图片路径
	//savePath 换脸后想要保存的路径
	//return 换脸后实际保存的路径
	public native String Calculate(String srcPath,String dstPath, String savaPath);
	
	
	//计算换脸
	//srcPath 上传的用户相片路径
	//dstPath 要被换脸的图片路径
	//maskPath 要被换脸的前景掩码图片路径
	//savePath 换脸后想要保存的路径
	//return 换脸后实际保存的路径
	public native String CalculateWithMask(String srcPath,String dstPath,String maskPath, String savaPath);
}
