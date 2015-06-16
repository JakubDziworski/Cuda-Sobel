
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


#define PRIMARY_THRESHOLD 125
#define SECONDARY_THRESHOLD 75


__device__ int sobel(int a, int b, int c, int d, int e, int f) {
	return ((a + 2 * b + c) - (d + 2 * e + f));
}

__device__ unsigned char
ComputeSobel(unsigned char ul, // upper left
unsigned char um, // upper middle
unsigned char ur, // upper right
unsigned char ml, // middle left
unsigned char mm, // middle (unused)
unsigned char mr, // middle right
unsigned char ll, // lower left
unsigned char lm, // lower middle
unsigned char lr, // lower right
float fScale)
{
	short Horz = ur + 2 * mr + lr - ul - 2 * ml - ll;
	short Vert = ul + 2 * um + ur - ll - 2 * lm - lr;
	short Sum = (short)(fScale*(abs((int)Horz) + abs((int)Vert)));

	if (Sum < 125)
	{
		return 0;
	}
	else if (Sum > 125)
	{
		return 255;
	}

	return (unsigned char)Sum;
}

__device__ int getindexForPixelAt(unsigned char *pixels, int x, int y, int width, int height)
{
	int val = 4 * (x + y*width);
	const int max = width*height * 4-1;
	if (val < 0)
	{
		//printf("\nx %d: y %d: below 0",x,y);
		return 0;
	}
	if (val > max)
	{
		//printf("\nx %d: y %d : above max", x, y);
		return max;
	}
	return val;
}

__device__ float4 pixelAt(unsigned char *pixels, int x, int y, int width,int height)
{
	int start = getindexForPixelAt(pixels, x, y, width, height);
	return float4{ pixels[start], pixels[start + 1], pixels[start + 2], pixels[start + 3] };
}



__global__ void addKernel(unsigned char *pixels,int width,int height)
{
	int id = blockIdx.x*blockDim.x+ threadIdx.x;
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (y >= height-1 || x >= width-1) return;
		float4 ul, um, ur, ml, mm, mr, ll, lm, lr;

		ul = pixelAt(pixels, x - 1, y + 1, width, height);
		um = pixelAt(pixels, x, y + 1, width, height);
		ur = pixelAt(pixels, x + 1, y + 1, width, height);

		ml = pixelAt(pixels, x - 1, y, width, height);
		mm = pixelAt(pixels, x, y, width, height);
		mr = pixelAt(pixels, x+1, y, width, height);

		ll = pixelAt(pixels, x -1, y - 1, width, height);
		lm = pixelAt(pixels, x, y - 1, width, height);
		lr = pixelAt(pixels, x+1, y - 1, width, height);

		int pos = getindexForPixelAt(pixels, x, y, width, height);
		pixels[pos] = ComputeSobel(ul.x, um.x, ur.x, ml.x, mm.x, mr.x, ll.x, lm.x, lr.x, 1.0f);
		pixels[pos+1] = ComputeSobel(ul.y, um.y, ur.y, ml.y, mm.y, mr.y, ll.y, lm.y, lr.y, 1.0f);
		pixels[pos+2] = ComputeSobel(ul.z, um.z, ur.z, ml.z, mm.z, mr.z, ll.z, lm.z, lr.z, 1.0f);
		pixels[pos + 3] = ComputeSobel(ul.w, um.w, ur.w, ml.w, mm.w, mr.w, ll.w, lm.w, lr.w, 1.0f);
}

extern "C" void initFilter(unsigned char *pixels, int width, int height)
{
	const unsigned int size = width*height*4;

	unsigned char *d_pixels;
	cudaMalloc(&d_pixels, size);
	cudaMemcpy(d_pixels, pixels, size, cudaMemcpyHostToDevice);


	const int blockDim = 4;
	dim3 threads(blockDim, blockDim);
	dim3 blocks(width / blockDim + 1, height / blockDim + 1);

	printf("\nthreads in block = %d,blocks = %d", threads.x*threads.y, blocks.x*blocks.y);
	addKernel << < blocks, threads >> >(d_pixels, width, height);
	if (cudaSuccess != cudaGetLastError())
	{
		printf("kernel error %s", cudaGetLastError());
	}
	printf("\nsizeof size = %d sizeof dev =%d ", sizeof(*d_pixels), sizeof(*pixels));
	gpuErrchk(cudaMemcpy(pixels, d_pixels, size, cudaMemcpyDeviceToHost));
	
	cudaFree(d_pixels);
}


