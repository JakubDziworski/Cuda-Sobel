
#include <windows.h>
#include <glew.h>
#include <freeglut.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <stdlib.h>
#include <stdio.h>
#include <SOIL.h>
#include <string>
#include <cuda.h>

extern "C" void initFilter(unsigned char *pixels, int width, int height);
void initGL();
void prepareTexture(bool);

GLuint img;
GLubyte *original;

std::string ExePath() 
{
	char buffer[MAX_PATH];
	GetModuleFileName(NULL, buffer, MAX_PATH);
	std::string::size_type pos = std::string(buffer).find_last_of("\\/");
	return std::string(buffer).substr(0, pos);
}

void loadImageData()
{
	// load image (needed so we can get the width and height before we create the window
	std::string str = ExePath().append("\\..\\lena.png").c_str();
	img = SOIL_load_OGL_texture(
		str.c_str(),
		SOIL_LOAD_AUTO,
		SOIL_CREATE_NEW_ID,
		SOIL_FLAG_MIPMAPS | SOIL_FLAG_INVERT_Y | SOIL_FLAG_NTSC_SAFE_RGB | SOIL_FLAG_COMPRESS_TO_DXT);
	printf("%s ,%s", SOIL_last_result(), str);
	prepareTexture(true);
}


void display();
void keyboard(unsigned char key, int /*x*/, int /*y*/);
void reshape(int x, int y);
void timerEvent(int value);

void main(int argc, char **argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_SINGLE | GLUT_DEPTH);
	glutInitWindowSize(768, 768);
	glutCreateWindow("CUDA Filter");
	glutDisplayFunc(display);

	loadImageData();

	glutKeyboardFunc(keyboard);
	glutReshapeFunc(reshape);
	glutTimerFunc(10/*ms*/, timerEvent, 0);
	glutMainLoop();
}
void prepareTexture(bool filtering)
{
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, img);
	int height, width;
	glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH , &width);
	glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &height);
	printf("\nTexture size %d x %d", width, height);
	const int size = width*height * 4;
	GLubyte *pixels = new GLubyte[size];
	glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
	if (original == NULL)
	{
		original = new unsigned char[size];
		memcpy(original, pixels, size);
	}
	
	if (filtering)
	{
		initFilter(pixels, width, height);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
		std::string path = ExePath().append("\\..\\converted.bmp");
		int result = SOIL_save_image(path.c_str(), SOIL_SAVE_TYPE_BMP, width, height, GL_RGBA, pixels);
		printf("\nsaved to file : %s . result : %d", path, result);
	}
	else 
	{
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, original);
	}

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glutSwapBuffers();
}



void display()
{
	glBegin(GL_QUADS);
	glTexCoord2d(0, 0);
	glVertex2d(-1,-1);

	glTexCoord2d(1, 0);
	glVertex2d(1, -1);

	glTexCoord2d(1, 1);
	glVertex2d(1,1);

	glTexCoord2d(0, 1);
	glVertex2d(-1, 1);
	glEnd();
	glutSwapBuffers();
}

void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
	if (key == 'e')
	{
		prepareTexture(true);
	}
	if (key == 'd')
	{
		prepareTexture(false);
	}
}

void reshape(int x, int y)
{

}

void timerEvent(int value)
{

}