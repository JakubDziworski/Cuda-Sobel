#ifndef __UTILS__H___
#define __UTILS__H___

#include <windows.h>
#include <stdlib.h>
#include <stdio.h>
#include <SOIL.h>
#include <string>
#include <freeglut.h>

std::string ExePath()
{
	char buffer[MAX_PATH];
	GetModuleFileName(NULL, buffer, MAX_PATH);
	std::string::size_type pos = std::string(buffer).find_last_of("\\/");
	return std::string(buffer).substr(0, pos);
}

GLuint loadImageData(std::string path)
{
	// load image (needed so we can get the width and height before we create the window
	std::string str = ExePath().append(path).c_str();
	GLuint img = SOIL_load_OGL_texture(
		str.c_str(),
		SOIL_LOAD_AUTO,
		SOIL_CREATE_NEW_ID,
		SOIL_FLAG_MIPMAPS | SOIL_FLAG_INVERT_Y | SOIL_FLAG_NTSC_SAFE_RGB | SOIL_FLAG_COMPRESS_TO_DXT);
	printf("%s ,%s", SOIL_last_result(), str);
	return img;
}
#endif