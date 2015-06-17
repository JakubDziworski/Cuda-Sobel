
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = false)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "\nGPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__device__ int sobel(int a, int b, int c, int d, int e, int f) {
	return ((a + 2 * b + c) - (d + 2 * e + f));
}

__device__ int getindexForPixelAt(unsigned char *pixels, int x, int y, int width, int height)
{
	int val = 4 * (x + y*width);
	const int max = width*height * 4-1;
	if (val < 0)
	{
		return 0;
	}
	if (val > max)
	{
		return max;
	}
	return val;
}

__device__ float4 pixelAt(unsigned char *pixels, int x, int y, int width,int height)
{
	int start = getindexForPixelAt(pixels, x, y, width, height);
	return float4{ pixels[start], pixels[start + 1], pixels[start + 2], pixels[start + 3] };
}



__global__ void sobelKernel(unsigned char *pixels,unsigned char *originalPixels,int width,int height)
{
	int id = blockIdx.x*blockDim.x+ threadIdx.x;
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (y >= height-1 || x >= width-1) return;
		float4 x0, x1, x2, x3, x4, x5, x6, x7, x8;
		//gora
		x0 = pixelAt(originalPixels, x - 1, y - 1, width, height);
		x1 = pixelAt(originalPixels, x, y - 1, width, height);
		x2 = pixelAt(originalPixels, x + 1, y - 1, width, height);
		//srodek
		x3 = pixelAt(originalPixels, x - 1, y, width, height);
		x4 = pixelAt(originalPixels, x, y, width, height);
		x5 = pixelAt(originalPixels, x + 1, y, width, height);
		//dol
		x6 = pixelAt(originalPixels, x - 1, y + 1, width, height);
		x7 = pixelAt(originalPixels, x, y + 1, width, height);
		x8 = pixelAt(originalPixels, x + 1, y + 1, width, height);

		int dfdy_r = sobel(x6.x, x7.x, x8.x, x0.x, x1.x, x2.x);
		int dfdx_r = sobel(x2.x, x5.x, x8.x, x0.x, x3.x, x6.x);

		int dfdy_g = sobel(x6.y, x7.y, x8.y, x0.y, x1.y, x2.y);
		int dfdx_g = sobel(x2.y, x5.y, x8.y, x0.y, x3.y, x6.y);

		int dfdy_b = sobel(x6.z, x7.z, x8.z, x0.z, x1.z, x2.z);
		int dfdx_b = sobel(x2.z, x5.z, x8.z, x0.z, x3.z, x6.z);

		int gradient_r = abs(dfdy_r) + abs(dfdx_r);
		int gradient_g = abs(dfdy_g) + abs(dfdx_g);
		int gradient_b = abs(dfdy_b) + abs(dfdx_b);

		float gradient = (gradient_r + gradient_g + gradient_b) / 3.0f;

		float4 new_pixel = float4{ gradient, gradient, gradient, 255 };
		
		int pos = getindexForPixelAt(originalPixels, x, y, width, height);
		pixels[pos] = new_pixel.x;
		pixels[pos + 1] = new_pixel.y;
		pixels[pos + 2] = new_pixel.z;
}

extern "C" void sobelFilter(unsigned char *pixels, int width, int height)
{
	const unsigned int size = width*height*4;

	unsigned char *d_pixels;
	cudaMalloc(&d_pixels, size);
	cudaMemcpy(d_pixels, pixels, size, cudaMemcpyHostToDevice);

	unsigned char *d_original_pixels;
	cudaMalloc(&d_original_pixels, size);
	cudaMemcpy(d_original_pixels, pixels,size, cudaMemcpyHostToDevice);

	const int blockDim = 16;
	dim3 threads(blockDim, blockDim);
	dim3 blocks(width / blockDim, height / blockDim);
	printf("\nthreads in block = %d,blocks = %d", threads.x*threads.y, blocks.x*blocks.y);

	sobelKernel << < blocks, threads >> >(d_pixels, d_original_pixels, width, height);
	if (cudaSuccess != cudaGetLastError())
	{
		printf("\nKernel error %s", cudaGetLastError());
	}

	gpuErrchk(cudaMemcpy(pixels, d_pixels, size, cudaMemcpyDeviceToHost));
	
	cudaFree(d_original_pixels);
	cudaFree(d_pixels);
}


