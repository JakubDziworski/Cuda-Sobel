/*
* Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

#ifndef __SOBELFILTER_KERNELS_H_
#define __SOBELFILTER_KERNELS_H_


// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define checkCudaErrors(err)           __checkCudaErrors (err, __FILE__, __LINE__)

inline void __checkCudaErrors(cudaError err, const char *file, const int line)
{
	if (cudaSuccess != err)
	{
		fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",
			file, line, (int)err, cudaGetErrorString(err));
		//exit(EXIT_FAILURE);
	}
}

typedef unsigned char Pixel;

// global determines which filter to invoke
enum SobelDisplayMode
{
	SOBELDISPLAY_IMAGE = 0,
	SOBELDISPLAY_SOBELTEX,
	SOBELDISPLAY_SOBELSHARED
};


extern enum SobelDisplayMode g_SobelDisplayMode;

extern "C" void setupTexture(int iw, int ih, Pixel *data);
extern "C" void deleteTexture(void);

#endif

