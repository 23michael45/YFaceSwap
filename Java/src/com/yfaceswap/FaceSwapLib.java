package com.yfaceswap;
public class FaceSwapLib {
	
	
	//��ʼ�����������
	public native boolean Init();
	
	//���ٻ��������
	public native boolean Finalize();
	
	//���㻻��
	//srcPath �ϴ����û���Ƭ·��
	//dstPath Ҫ��������ͼƬ·��
	//savePath ��������Ҫ�����·��
	//return ������ʵ�ʱ����·��
	public native String Calculate(String srcPath,String dstPath, String savaPath);
}
