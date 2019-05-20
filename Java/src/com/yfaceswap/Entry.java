package com.yfaceswap;
public class Entry {
	
	
	 public static void main(String[] args) {
		 
	        //System.loadLibrary("WindowsLib");
	        // System.loadLibrary("YFaceSwapLib");
			System.load("/usr/lib/jvm/java-8-openjdk-amd64/lib/libYFaceSwapLib.so");
			
	        FaceSwapLib fslib = new FaceSwapLib();
			// String detectModelPath = "";
			// String alignModelPath = "";

			
			String detectModelPath = "/usr/lib/jvm/java-8-openjdk-amd64/lib/haarcascade_frontalface_default.xml";			
			String alignModelPath = "/usr/lib/jvm/java-8-openjdk-amd64/lib/shape_predictor_68_face_landmarks.dat"; 
			if(!fslib.Init(detectModelPath,alignModelPath))
			{

				System.out.println("Init Failed");
				return;
			}
			System.out.println("Init Success");
	        
			String srcPath = "/root/DevelopProj/Yuji/YFaceSwap/bin/upload.jpg";//"images/src/1.jpg";
			String dstPath = "/root/DevelopProj/Yuji/YFaceSwap/bin/dst.png";//"images/dst/1.jpg";
			String savePath = "/root/DevelopProj/Yuji/YFaceSwap/bin/upload_dst.jpg";//"images/save/1_1.jpg";
			

			String retPath = fslib.Calculate(srcPath,dstPath,savePath);
			
			if(retPath == "")
			{
				System.out.println("Input File or Folder not found");
				return;
			}
	        System.out.println("Calcuate Success :" + retPath);
			
			fslib.Finalize();
			
	        System.out.println("Finish");
	 }
	 
	 
}
