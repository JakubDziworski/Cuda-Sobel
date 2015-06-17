
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
#include "utils.h"

extern "C" void sobelFilter(unsigned char *pixels, int width, int height);
void initGL();
void prepareTexture();

GLuint img;

void display();
void keyboard(unsigned char key, int /*x*/, int /*y*/);
void reshape(int x, int y);
void timerEvent(int value);

void main(int argc, char **argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA);
	glutInitWindowSize(768, 768);
	glutCreateWindow("CUDA sobel Filter - krawedzie");
	glutDisplayFunc(display);

	img = loadImageData("\\..\\1024.jpg");
	prepareTexture();

	glutKeyboardFunc(keyboard);
	glutReshapeFunc(reshape);
	glutTimerFunc(10/*ms*/, timerEvent, 0);
	glutMainLoop();
}
void prepareTexture()
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
	//cuda fun
	sobelFilter(pixels, width, height);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixels);

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
}

void reshape(int x, int y)
{

}

void timerEvent(int value)
{

}