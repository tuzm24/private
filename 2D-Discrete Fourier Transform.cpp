#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#pragma warning(disable:4996)

typedef unsigned char UChar;
typedef double Double;

#define IMAGE_WIDTH 256
#define IMAGE_HEIGHT 256
#define IMAGE_AREA ((IMAGE_WIDTH)*(IMAGE_HEIGHT))
#define IMAGE_PATH "lena.img"
#define TWOPI ((3.141592)*(2))
#define BIT_DEPTH 8


// Filtering Option
#define LOWPASS_IN_FREQ 0
#define HIGHPASS_IN_FREQ 1
#define STRONG_OF_FILTER 0.2

typedef struct Image {
	UChar *img;
	int width;
	int height;

}Image;


typedef struct Complex {
	double real;
	double image;
}Complex;

typedef struct FreqImge {
	Complex *img;
	int width;
	int height;
}FreqImage;

UChar* memoryAlloc(int width, int height) {
	return (UChar*)calloc(width*height, sizeof(UChar));
}

FreqImage* initFreqImage(int width, int height) {
	FreqImage *img = (FreqImage*)malloc(sizeof(FreqImage));
	img->img = (Complex*)calloc(width*height, sizeof(Complex));
	img->width = width;
	img->height = height;
	return img;
}


Image* ImageInit(int width, int height) {
	Image *img = (Image*)malloc(sizeof(Image));
	img->img = memoryAlloc(width, height);
	img->width = width;
	img->height = height;
	return img;
}

//_inline double isOdd(int x) {
//	if (x % 2 == 1)
//		return -1;
//	return 1;
//}

void ImageWrite(Image *img, const char filename[]) {
	FILE *fp = fopen(filename, "wb");
	fwrite(img->img, img->width*img->height, sizeof(UChar), fp);
	fclose(fp);
	return;
}

Double Clip3(int p) {
	int max_range = (1 << BIT_DEPTH) - 1;
	if (p > max_range)
		return max_range;
	else if (p < 0)
		return 0;
	else
		return p;

}



void FreqMagnitudeImageShow(FreqImage *img) {
	int width = img->width, height = img->height;
	Complex *dcvalue = img->img + width*height/2+width/2;
	double dc_coef = log10(sqrt(pow(dcvalue->real, 2) + pow(dcvalue->image, 2)) + 1);
	Image *coef_img = NULL;
	coef_img = ImageInit(width, height);
	UChar *coef = coef_img->img;
	int bitmax = (1 << BIT_DEPTH) - 1;
	Complex *temp = img->img;
	for(int i=0; i<height; i++) 
		for (int j = 0; j < width; j++) {
			coef[i*width + j] = Clip3(((double)bitmax * (log10(sqrt(pow(temp->real, 2) + pow(temp->image, 2)) + 1) / dc_coef))+0.5);
			temp++;
		}
	ImageWrite(coef_img, "FreqDomain.img");
	free(coef);
	free(coef_img);
	return;
}


FreqImage* DiscreteFourierTrans(Image *img) {
	FreqImage *fimg;
	double sub, intri;
	int width = img->width, height = img->height;
	int area = width * height;
	fimg = initFreqImage(width, height);
	Complex *temp = fimg->img;
	double two_pi_div = TWOPI / (height*width); // 실수형 연산을 줄이기 위해 미리 연산
	UChar *img_addr;
	for(int u= 0; u<height; u++)
		for (int v = 0; v < width; v++) {
			img_addr = img->img;
			for(int m=0; m<height; m++)
				for (int n = 0; n < width; n++) {
					sub = ((m + n) & 1? -1 : 1) * (int)(*img_addr); // 연산 가속화를 위해 비트연산을 통한 삼항연산자.... 삼각함수 안에 들어갈 값을 한번만 미리 계산
					intri = two_pi_div * (u*m*width + v*n*height); // 실수연산을 1번으로 줄임
					temp->real += sub * cos(intri);
					temp->image -= sub * sin(intri);
					img_addr++;
				}
			temp->real = (temp->real)/area;
			temp->image = (temp->image)/area;
			++temp;
		}
	printf("DFT Complete\n");
	FreqMagnitudeImageShow(fimg);
	return fimg;
}


void InverseDiscreteFourierTrans(FreqImage *fimg) {
	int width = fimg->width, height = fimg->height;
	Image *recon_img = NULL;
	recon_img = ImageInit(width, height);
	UChar *temp = recon_img->img;
	Complex *ftmp;
	double intri;
	double two_pi_div = TWOPI / (width*height);
	double tmp;
	for(int m=0; m<height; m++)
		for (int n = 0; n < width; n++) {
			ftmp = fimg->img;
			tmp = 0;
			for(int u=0; u<height; u++)
				for (int v = 0; v < width; v++) {
					intri = two_pi_div * (u*m*width + v*n*height);
					tmp += ((m + n) & 1 ? -1 : 1)*(cos(intri)*(ftmp->real) - sin(intri)*(ftmp->image));//
					ftmp++;
				}
#if HIGHPASS_IN_FREQ
			tmp = Clip3(tmp);
#endif
			*temp = tmp;
			temp++;
		}
	ImageWrite(recon_img, "Recon.img");
	free(recon_img->img);
	free(recon_img);
}



void FilteringInFreqDomain(FreqImage *fimg) {
	int width = fimg->width, height = fimg->height;
	int width_d = width / 2, height_d = height / 2;
	int min;
	width < height ? min = width_d : min = height_d;
	int threshold = pow(min * STRONG_OF_FILTER,2);
	Complex *temp = fimg->img;
	for(int u=0; u<height; u++)
		for (int v = 0; v < width; v++) {
#if LOWPASS_IN_FREQ
			((u - height_d)*(u - height_d) + (v - width_d)*(v - width_d)) <= threshold ? 0 : temp->real = temp->image = 0;
#endif
#if HIGHPASS_IN_FREQ
			((u - height_d)*(u - height_d) + (v - width_d)*(v - width_d)) >= threshold ? 0 : temp->real = temp->image = 0;
#endif
			
			temp++;
		}
	return;

}


int main() {
	Image *img = NULL;
	img = ImageInit(IMAGE_WIDTH, IMAGE_HEIGHT);
	FILE *fp;
	FreqImage *transimg;
	fp = fopen(IMAGE_PATH, "rb");
	fread(img->img, sizeof(UChar), img->height*img->width, fp);
	transimg = DiscreteFourierTrans(img);
#if LOWPASS_IN_FREQ||HIGHPASS_IN_FREQ
	FilteringInFreqDomain(transimg);
#endif
	InverseDiscreteFourierTrans(transimg);
	fclose(fp);
	free(img->img);
	free(img);
	free(transimg->img);
	free(transimg);
	return 0;
}