
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cufft.h"

#include <stdio.h>
#include <opencv2/opencv.hpp>
#ifdef _DEBUG
#pragma comment(lib,"opencv_world411d.lib")
#else
#pragma comment(lib,"opencv_world411.lib")
#endif 
#pragma comment(lib,"cufft.lib")


const int w = 1440, h = 1080;
const int size = 2048;

cufftHandle fftPlan;
cufftResult fresu;


__device__ unsigned char getr(float x) {
	return (tanh((x - 0.375f) * 6.0f) + 1.0f) * 127.0f;
}
__device__ unsigned char getg(float x) {
	return (tanh((x - 0.6250f) * 6.0f) + 1.0f) * 127.0f;
}
__device__ unsigned char getb(float x) {
	return (exp(-20.0f * (x - 0.25f) * (x - 0.25f) - 2.0f * exp(-(x + 0.05f) * (x + 0.05f) * 144.0f)) * 0.5f + 1.0f + tanh((x - 0.875f) * 6.0f)) * 127.0f;
}

__global__ void imgfill(float2* d_k, uchar3* d_img) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int imgx, imgy;
	imgx = (x >= size / 2) ? x - size / 2 : x + size / 2;
	imgy = (y >= size / 2) ? y - size / 2 : y + size / 2;
	float2 k = d_k[y * size + x];
	float in = k.x * k.x + k.y * k.y;
	in = log(in * (1.0f / 256.0f/size) + 0.8f) * 0.07f;
	uchar3 c;
	c.x = getb(in);
	c.y = getg(in);
	c.z = getr(in);
	d_img[imgy * size + imgx] = c;
}

__global__ void fill(float2* d_x, uchar3* d_8uc3) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int imgx, imgy;
	float cx, cy;
	unsigned char r;
	if (x >= size / 2 + w / 2) {
		imgx = 0;
		cx = size - x;
		cx = exp(-cx * cx * (1.0f / 1024.0f));
	}
	else if (x < size / 2 + w / 2 && x >= w) {
		imgx = w - 1;
		cx = x - w;
		cx = exp(-cx * cx * (1.0f / 1024.0f));
	}
	else {
		imgx = x;
		cx = 1.0f;
	}

	if (y >= size / 2 + h / 2) {
		imgy = 0;
		cy = size - y;
		cy = exp(-cy * cy * (1.0f / 1024.0f));
	}
	else if (y < size / 2 + h / 2 && y >= h) {
		imgy = h - 1;
		cy = y - h;
		cy = exp(-cy * cy * (1.0f / 1024.0f));
	}
	else {
		imgy = y;
		cy = 1.0f;
	}

	r = d_8uc3[imgy * w + imgx].x;
	d_x[y * size + x].x = r * cx * cy;
	d_x[y * size + x].y = 0;
}

int main()
{
	cv::VideoCapture vc("C:/files/avg/badapple.mp4");
	cv::Mat frame(h, w, CV_8UC3, cv::Scalar(0, 0, 0));
	//cv::Mat frame1(1080, 1440, CV_8UC3, cv::Scalar(0, 0, 0));
	cv::Mat ff(size, size, CV_8UC3, cv::Scalar(0, 0, 0));
	cv::VideoWriter writer;
	writer.open("C:/files/avg/badapplek.mp4", cv::VideoWriter::fourcc('a', 'v', 'c', '1'), 30.0, cv::Size(size, size));
	cufftPlan2d(&fftPlan, size, size, CUFFT_C2C);

	float2* d_x, * d_k;
	uchar3* d_8uc3, * d_img;

	cudaMalloc(&d_x, w * 2 * h * 2 * sizeof(float2));
	cudaMalloc(&d_k, w * 2 * h * 2 * sizeof(float2));
	cudaMalloc(&d_8uc3, w * h * 3);
	cudaMalloc(&d_img, w * 2 * h * 2 * 3);

	for (int i = 0; i < (3 * 60 + 40) * 30 - 40; i++) {
		vc >> frame;
		cudaMemcpy(d_8uc3, frame.data, w * h * 3, cudaMemcpyHostToDevice);
		fill << < dim3(size / 128, size, 1), dim3(128, 1, 1) >> > (d_x, d_8uc3);
		cufftExecC2C(fftPlan, d_x, d_k, CUFFT_FORWARD);
		imgfill << < dim3(size / 128, size, 1), dim3(128, 1, 1) >> > (d_k, d_img);
		cudaMemcpy(ff.data, d_img, size * size * 3, cudaMemcpyDeviceToHost);
		writer << ff;
		printf("%f\n", i * 1.0 / (3 * 60 + 40) / 30);
	}

	return 0;
}
