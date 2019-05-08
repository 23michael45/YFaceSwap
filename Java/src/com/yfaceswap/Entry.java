package com.facecloud;
public class Entry {
	
	
	 public static void main(String[] args) {
		 
	        //System.loadLibrary("WindowsLib");
	        System.loadLibrary("FaceSwapLib");
	        
	        
	        FaceSwapLib fslib = new FaceSwapLib();
	        
	        fslib.Init();
	        
			String srcPath = "images/src/1.jpg";
			String dstPath = "images/dst/1.jpg";
			String savePath = "images/save/1_1.jpg";
			

			String retPath = fslib.Calculate(srcPath,dstPath,savePath);
			
	        System.out.println("Calcuate OK");
			
			fslib.Finalize();
			
	        System.out.println("Finish");
	 }
	 
	 
}
