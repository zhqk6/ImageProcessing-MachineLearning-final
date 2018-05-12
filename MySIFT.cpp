#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <vector>
#include <iostream>
#include <windows.h>
#include <stdio.h>

#if 0
typedef short sift_wt;
static const int SIFT_FIXPT_SCALE = 48;
#else
typedef float sift_wt;
static const int SIFT_FIXPT_SCALE = 1;
#endif
static const float SIFT_INIT_SIGMA = 0.5f;
//sift_wt is a specific class 

using namespace cv;
using namespace std;
const int s = 3;
//In Lowe's paper, s=3 is suggest 
const float first_sigma = 1.6f;
//sigma for the first image

Mat Base_img(const Mat& img, float sigma)
{
	//create the first blurred image at the first octave
	//img is input image
	//sigma is the parameter of the first Gaussian
	//returns the first blurred image at the first octave

	Mat BaseImage;
	img.copyTo(BaseImage);
	BaseImage.convertTo(BaseImage, DataType<sift_wt>::type, SIFT_FIXPT_SCALE, 0);
	float variance_sigma;
	variance_sigma = sqrtf(max(sigma * sigma - 1 * 0.5f, 0.01f));
	//base sigma here is 1.6
	GaussianBlur(BaseImage, BaseImage, Size(), variance_sigma, variance_sigma);
	//Gaussian blur to the first image
	return BaseImage;
}

void CreateDOG(const Mat& base, vector<Mat>& LOGpyr, vector<Mat>& DOGpyr, int num_octaves)
{
	//create DOG pyramid
	//base is the first blurred image at the first octave
	//LOGpyr is LOG pyramid
	//DOGpyr is DOG pyramid
	//num_octaves is the number of octaves

	vector<double> sigma(s + 3);
	LOGpyr.resize(num_octaves*(s + 3));
	int nOctaves = (int)LOGpyr.size() / (s + 3);
	DOGpyr.resize(nOctaves*(s + 2));
	sigma[0] = first_sigma;

	double k = pow(2.0, 1.0 / s);
	//k=2^(1/s)
	for (int i = 1; i < s + 3; i++)
	{
		double sig_prev = pow(k, (double)(i - 1))*first_sigma;
		double sig_total = sig_prev*k;
		sigma[i] = sqrt(sig_total*sig_total - sig_prev*sig_prev);
	}
	//calculating different sigma values
	//sigma=(k^r)*first_sigma

	for (int i = 0; i < num_octaves; i++)
	{
		for (int j = 0; j < s + 3; j++)
		{
			Mat& dst = LOGpyr[i*(s + 3) + j];
			if (i == 0 && j == 0)
				dst = base;
			//the first blurred image
			else if (j == 0)
			{
				const Mat& src = LOGpyr[(i - 1)*(s + 3) + s];
				resize(src, dst, Size(src.cols / 2, src.rows / 2),
					0, 0, INTER_NEAREST);
				//next octave is made by downsampling last octave
			}
			else
			{
				const Mat& src = LOGpyr[i*(s + 3) + j - 1];
				GaussianBlur(src, dst, Size(), sigma[j], sigma[j]);
				//Gaussian blur with different sigma
			}
		}
	}
	for (int i = 0; i < nOctaves; i++)
	{
		for (int j = 0; j < s + 2; j++)
		{
			const Mat& image1 = LOGpyr[i*(s + 3) + j];
			const Mat& image2 = LOGpyr[i*(s + 3) + j + 1];
			Mat& result = DOGpyr[i*(s + 2) + j];
			subtract(image1, image2, result, noArray(), DataType<sift_wt>::type);
			//Creating DOG pyramid by subtraction between 2 LOG image
		}
	}
}

float Gradient_Orientation(const Mat& img, Point location, int radius,
	float sigma, float* hist, int n)
{
	//Calculating the histogram of gradient of each point
	//location contains x,y
	//radius is the radius of circular window
	//sigma is current sigma
	//hist is the histogram to save
	//n=36 here because 10 degrees for 1 bin

	int i, j, k, num_pixel = (radius * 2 + 1)*(radius * 2 + 1);
	//num_pixel is the number of pixels in circular window around the keypoint
	float exp_num = -1.f / (2.f * sigma * sigma);
	// exp_num is Gaussian weight number e

	AutoBuffer<float> tem_buffer(num_pixel * 4 + n + 4);
	float *X = tem_buffer, *Y = X + num_pixel, *Magnitude = X, *Orientation = Y + num_pixel, *W = Orientation + num_pixel;
	float* tem_hist = W + num_pixel + 2;
	// X,Y is dx,dy
	// W is Gaussian weight
	//gradient magnitude share with X
	//Orientation is gradient orientation 
	//temphist save the histogram temporarily£¬length is 38£¬above W
	//length is 38 because we need to do circulation later

	for (i = 0; i < n; i++) {
		tem_hist[i] = 0.f;
	}
	//clear the histogram

	for (i = -radius, k = 0; i <= radius; i++)
	{
		int y = location.y + i;
		//y location of each pixel
		if (y <= 0 || y >= img.rows - 1)
			continue;
		for (j = -radius; j <= radius; j++)
		{
			int x = location.x + j;
			//x location of each pixel
			if (x <= 0 || x >= img.cols - 1)
				continue;
			float dx = (float)(img.at<sift_wt>(y, x + 1) - img.at<sift_wt>(y, x - 1));
			float dy = (float)(img.at<sift_wt>(y - 1, x) - img.at<sift_wt>(y + 1, x));
			//calculating dx, dy
			X[k] = dx; Y[k] = dy; W[k] = (i*i + j*j)*exp_num;
			//save values
			k++;
			//k is the number of pixels in circular window
		}
	}
	num_pixel = k;
	exp(W, W, num_pixel);
	fastAtan2(Y, X, Orientation, num_pixel, true);
	magnitude(X, Y, Magnitude, num_pixel);
	//calculating all pixels' Gaussian weight W, gradient orientation and gradient magnitude
	for (k = 0; k < num_pixel; k++)
	{
		int bins = cvRound((n / 360.f)*Orientation[k]);
		//which gradient orientation it belongs to
		if (bins >= n)
			bins -= n;
		if (bins < 0)
			bins += n;
		//using circulation if beyond the range
		tem_hist[bins] += W[k] * Magnitude[k];
		//after weighted
	}
	tem_hist[-1] = tem_hist[n - 1];
	tem_hist[-2] = tem_hist[n - 2];
	tem_hist[n] = tem_hist[0];
	tem_hist[n + 1] = tem_hist[1];
	//preparing the first and last one in advance because we need to process as a circulation later
	for (i = 0; i < n; i++)
	{
		hist[i] = (tem_hist[i - 2] + tem_hist[i + 2])*(1.f / 16.f) +
			(tem_hist[i - 1] + tem_hist[i + 1])*(4.f / 16.f) +
			tem_hist[i] * (6.f / 16.f);
	}
	//H(i)=(h(i-2)+h(i+2))/16+4*(h(i-1)+h(i+1))/16+6*h(i)/16
	//h represents the histogram before smooth, H represents the histogram after smooth
	// smooth the histogram
	float Peak = hist[0];
	for (i = 1; i < n; i++)
		Peak = max(Peak, hist[i]);
	//calculating and returning the maximun of histogram
	return Peak;
}

void Hist_Show(Mat& hist, string&  winname)
{
	//Show the histogram of orientation of each keypoint
	//hist is the histogram created by Gradient_Orientation
	//winnname is the name of window

	Mat drawHist;
	int Hist_rows = hist.rows;
	int wide_hist = 360; 
	int height_hist = 360;
	//the wide and height of histogram
	int bin_w = cvRound((double)wide_hist / Hist_rows);
	//wide of one bin
	Mat histImage(wide_hist, height_hist, CV_8UC3, Scalar(0, 0, 0));
	//Create hist which will show
	normalize(hist, drawHist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	//Noramlize in the range [ 0, histImage.rows ]
	for (int i = 1; i < Hist_rows; i++)
	{
		rectangle(histImage, Point((i - 1)*bin_w, height_hist),
			Point(i*bin_w, height_hist - cvRound(drawHist.at<float>(i - 1))), Scalar(0, 0, 255), 1, 8, 0);
		// Represent as a rectangle
	}
	namedWindow(winname, 1);
	imshow(winname, histImage);
}

void findkeypoint(const vector<Mat>& LOGpyr, const vector<Mat>& DOGpyr, vector<KeyPoint>& keypoints) {
	//finding keypoints 
	//LOGpyr is LOG pyramid
	//DOGpyr is DOG pyramid
	//keypoints is parameter with KeyPoint class

	int nOctaves = (int)LOGpyr.size() / (s + 3);
	keypoints.clear();
	//clear keypoints
	int threshold = cvFloor(0.5*0.04 / s * 255);
	KeyPoint kpt;
	for (int i = 0; i < nOctaves; i++) {
		for (int j = 1; j < 4; j++) {
			Mat current_img = DOGpyr[i * 5 + j];
			Mat previous_img = DOGpyr[i * 5 + j - 1];
			Mat next_img = DOGpyr[i * 5 + j + 1];
			//in DOG, the neigbor layers in one octave
			int step = current_img.step1();
			for (int m = 5; m < current_img.rows-5; m++) {
				const sift_wt* current_pointer = current_img.ptr<sift_wt>(m);
				const sift_wt* previous_pointer = previous_img.ptr<sift_wt>(m);
				const sift_wt* next_pointer = next_img.ptr<sift_wt>(m);
				//row pointers
				for (int n = 5; n < current_img.cols-5; n++) {
					sift_wt pixel = current_pointer[n];
					if ((abs(pixel) > threshold) && ((pixel > 0 && pixel >= current_pointer[n - 1] && pixel >= current_pointer[n + 1] &&
						pixel >= current_pointer[n - step - 1] && pixel >= current_pointer[n - step] && pixel >= current_pointer[n - step + 1] &&
						pixel >= current_pointer[n + step - 1] && pixel >= current_pointer[n + step] && pixel >= current_pointer[n + step + 1] &&
						pixel >= next_pointer[n] && pixel >= next_pointer[n - 1] && pixel >= next_pointer[n + 1] &&
						pixel >= next_pointer[n - step - 1] && pixel >= next_pointer[n - step] && pixel >= next_pointer[n - step + 1] &&
						pixel >= next_pointer[n + step - 1] && pixel >= next_pointer[n + step] && pixel >= next_pointer[n + step + 1] &&
						pixel >= previous_pointer[n] && pixel >= previous_pointer[n - 1] && pixel >= previous_pointer[n + 1] &&
						pixel >= previous_pointer[n - step - 1] && pixel >= previous_pointer[n - step] && pixel >= previous_pointer[n - step + 1] &&
						pixel >= previous_pointer[n + step - 1] && pixel >= previous_pointer[n + step] && pixel >= previous_pointer[n + step + 1])
						||
						(pixel < 0 && pixel <= current_pointer[n - 1] && pixel <= current_pointer[n + 1] &&
							pixel <= current_pointer[n - step - 1] && pixel <= current_pointer[n - step] && pixel <= current_pointer[n - step + 1] &&
							pixel <= current_pointer[n + step - 1] && pixel <= current_pointer[n + step] && pixel <= current_pointer[n + step + 1] &&
							pixel <= next_pointer[n] && pixel <= next_pointer[n - 1] && pixel <= next_pointer[n + 1] &&
							pixel <= next_pointer[n - step - 1] && pixel <= next_pointer[n - step] && pixel <= next_pointer[n - step + 1] &&
							pixel <= next_pointer[n + step - 1] && pixel <= next_pointer[n + step] && pixel <= next_pointer[n + step + 1] &&
							pixel <= previous_pointer[n] && pixel <= previous_pointer[n - 1] && pixel <= previous_pointer[n + 1] &&
							pixel <= previous_pointer[n - step - 1] && pixel <= previous_pointer[n - step] && pixel <= previous_pointer[n - step + 1] &&
							pixel <= previous_pointer[n + step - 1] && pixel <= previous_pointer[n + step] && pixel <= previous_pointer[n + step + 1]))) {
						//Comparing each pixel to its 26 neighbors 

						int buffer[4];
						buffer[0] = n;
						buffer[1] = m;
						buffer[2] = j;
						buffer[3] = i;
						//buffer is to save informations.
						//n is cols, m is rows, j is layer, i is octave

						Mat originHist = Mat::zeros(36, 1, DataType<sift_wt>::type);
						float *hist = (float*)originHist.data;
						float omax = Gradient_Orientation(LOGpyr[i*(s + 3) + buffer[2]], Point(buffer[1], buffer[0]), cvRound(3 * 1.5* first_sigma), 1.5f*first_sigma, hist, 36);
						string winname = "Origin hist";
						Hist_Show(originHist, winname);
						waitKey(1000);
						//Generating histogram of gradient, can be commented

						for (int z = 0; z < 36; z++)
						{
							//z is the index of current bin£¬
							//pre is the index of previous bin£¬
							//next is the index of next bin
							int pre, next;
							if (z > 0) {
								pre = z - 1;
							}
							else {
								pre = 35;
							}
							if (z < 35) {
								next = z+1;
							}
							else {
								next = 0;
							}
								float bin = z + 0.5f * (hist[pre] - hist[next]) / (hist[pre] - 2 * hist[z] + hist[next]);
								if (bin < 0) {
									bin = 36 + bin;
								}
								else if (bin >= 36) {
									bin = bin - 36;
								}
								else {
									bin = bin;
								}
								//if they are beyond the range of bins[0-360 degrees], processing as a circulation
								kpt.pt.x = (float)buffer[1];
								kpt.pt.y = (float)buffer[0];
								kpt.octave = buffer[3] * (s + 3) + buffer[2];
								kpt.size = 1.6*powf(2.f, (float)(buffer[3] + buffer[2] / s));
								kpt.angle = 360.f - (float)((360.f / 36) * bin);
								//store informations of one keypoint.
								keypoints.push_back(kpt);
						}
					}
				}
			}
		}
	}
}

int main(int argc, char **argv)
{
	Mat img;
	img = imread("E:/ECE 8725/datasets/training/airplane/image(1).jpg", CV_LOAD_IMAGE_GRAYSCALE);
	Mat baseimage = Base_img(img, first_sigma);
	//Creating the first blurred image
	int number_octaves = cvRound(log((double)min(baseimage.cols, baseimage.rows)) / log(2.0) - 2);
	//number of octaves is round[log2(min(x,y)-2)], this image is 7
	vector<Mat> logpyr, dogpyr;
	CreateDOG(baseimage, logpyr, dogpyr, number_octaves);
	//Creating DOG pyramid

	char LOGfilename[100];
	char DOGfilename[100];
	for (int i = 0; i < (s+3)*number_octaves; i++) {
		sprintf(LOGfilename, "LOG(%d).jpg",i+1);
		imwrite(LOGfilename, logpyr[i]);
	}
	for (int i = 0; i < (s + 2)*number_octaves; i++) {
		sprintf(DOGfilename, "DOG(%d).jpg", i + 1);
		imwrite(DOGfilename, dogpyr[i]);
	}
	//save LOG and DOG images, comment it if don't need


	vector<KeyPoint> keypoints;
	findkeypoint(logpyr, dogpyr, keypoints);
	//find keypoints
	printf("size: %d\n", keypoints.size());
	for (int i = 0; i < keypoints.size(); i++) {
		float x = keypoints[i].pt.x;
		float y = keypoints[i].pt.y;
		int oct = keypoints[i].octave;
		int kptoctave = cvFloor(oct / 6);
		int layer = oct - 6 * kptoctave;
		float scale = (float)pow(2.0, (double)(0 - kptoctave));
		printf("keypointsx: %f\n", x);
		printf("keypointsy: %f\n", y);
		printf("keypointsoct: %d\n", oct);
		printf("scale: %f \n", scale);
		printf("angle: %f\n", keypoints[i].angle);
		printf("\n");
		Sleep(500);
	}
	printf("size: %d\n", keypoints.size());
	Sleep(20000);
	//printf the informations of each keypoint, comment it if don't need

	return 0;
}