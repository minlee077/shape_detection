#include <opencv2/opencv.hpp>

#include <iostream>
#include <cmath>
#include <vector>
#include <string>

#define RGBPIXEL2GREYSCALE(B,G,R) 0.3*R + 0.59*G + 0.11*B
#define BINARIZATION(P) ((P>125)?0:255)

using namespace cv;
using namespace std;

Mat convertRGB2Greyscale(Mat image);
Mat convertBinaryImage(Mat image,int &area);

void setContourStartPoint(Point& p, Mat image);
vector<Point> searchContour(Mat image);

double findPerpendicularDistance(Point p, Point p1, Point p2);
vector<Point> findVertex(vector<Point> v);
vector<Point> linePointDist(vector<Point>line, vector<Point> v);
double triangleArea(vector<Point>vertex);

void captionTriangle(Mat image,Mat& captionedImage);
void captionCircle(Mat image,Mat& captionedImage);

vector<Point> rdp(vector <Point>v, int epsilon);

const int captionTriangleArray[10][101] = {
			   { 1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1},
			   { 1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,1,1,1,0,0,0,0,0,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1},
			   { 1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,0,0,0,0,1,1,1,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,1,1,0,0,1,1,0,0,0,0,0,1,1,1,1,0,0,0,0,1,1,0,0,0,1,1,1,0,0,0,0,0,1,1,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0},
			   { 0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,1,0,0,0,1,1,1,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,1,1,0,0,1,1,0,0,0,0,0,1,1,0,1,1,0,0,0,1,1,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0},
			   { 0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,1,1,0,1,1,0,0,0,1,1,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1},
			   { 0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,1,1,0,0,0,0,1,1,0,0,1,1,0,0,1,1,0,0,0,1,1,1,0,0,1,1,1,1,1,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1},
			   { 0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,1,0,0,0,1,1,1,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,1,1,0,0,0,0,1,1,0,0,1,1,0,0,1,1,0,0,0,1,1,1,0,0,1,1,1,1,1,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0},
			   { 0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,1,0,0,0,0,1,1,1,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,1,1,1,0,0,0,0,1,1,1,0,0,0,1,1,0,0,0,1,1,0,1,1,0,0,0,1,1,1,0,0,0,0,0,1,1,0,0,0,1,1,1,0,0,0,0,0,0,1,0,0,0,1,1,1,0,0,0,0,0,0,0},
			   { 0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,1,0,0,0,0,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,0,0,0,0,1,1,1,0,0,0,1,1,0,0,0,0,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1},
			   { 0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,1,0,0,0,0,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,0,0,0,0,1,1,1,0,0,0,1,1,0,0,0,0,0,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1}
};

const int captionCircleArray[10][75] = {
			   { 1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1},
			   { 1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1},
			   { 1,1,1,0,0,0,0,0,1,1,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,1,1,1,0,0,0,0,1,1,1,0,0,0,1,1,1,0,0,0,0,0,1,1,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0},
			   { 1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,1,1,1,0,0,0,1,1,1,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0},
			   { 1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1},
			   { 1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1},
			   { 1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,1,1,1,0,0,0,1,1,1,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0},
			   { 1,1,1,0,0,0,0,0,1,1,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,1,1,1,0,0,0,0,1,1,1,0,0,0,1,1,1,0,0,0,0,0,1,1,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0},
			   { 1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,0,0,0,0,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1},
			   { 1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,0,0,0,0,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1}
};

int main()
{
	Mat image[4], greyScaleImage, binaryImage,captionedImage[4];
	vector<Point> contours,approx,vertex;

	int width, height,area=0;
	double estimatedArea = 0;
	for (int i = 0; i < 4; i++)
	{
		area = 0;
		String imagePath = "c:/input/"+ to_string(i+1) + ".jpg";
		image[i] = imread(imagePath);   // Read the file
		if (!image[i].data)                              // Check for invalid input
		{
			std::cout << "Could not open or find the image" << std::endl;
			return -1;
		}
		height = image[i].size().height;
		width = image[i].size().width;

		greyScaleImage = convertRGB2Greyscale(image[i]);// 그레이스케일변환
		binaryImage = convertBinaryImage(greyScaleImage, area); // 이진화, 이과정에서 도형에 해당하는 영역의 픽셀개수를 카운팅하여, 면적으로 지정한다.
		contours = searchContour(binaryImage); // 컨투어 탐색

		vertex = findVertex(contours); // 컨투어의 특정 점과 가장 먼 다른 점을 찾기 ( 삼각형의 경우 변 )
		vertex = linePointDist(vertex, contours); // 점과 직선사이 거리를 이용하여 직선과 가정먼 한점을 찾아냄 (삼각형의 경우 나머지 꼭짓점)

		estimatedArea = triangleArea(vertex); //세점을 통해 면적을구함.


		if ((estimatedArea / (double)area) > 0.9) // 이진화 과정에서 위에서 구한 삼각형 면적과 비교하여, 차이가 90프로 이하라면, 삼각형으로 판단.
		{

			captionTriangle(image[i], captionedImage[i]);
			cout << "Triangle" << endl;
		}
		else // 원인지 검증
		{
			approx = rdp(contours, 0.01 * contours.size()); //Ramer–Douglas–Peucker_algorithm을 적용하여, 컨투어를 단순화 한다.
			if (approx.size() > 8)// 단순화된 contour의 vertex가 일정개수 이상일 경우 원으로 판단. 
			{
				captionCircle(image[i], captionedImage[i]);
				cout << "Circle" << endl;//원 캡션;
			}
		}
	}

	cv::namedWindow("1.jpg Image", WINDOW_AUTOSIZE);// Create a window for display.
	cv::namedWindow("2.jpg Image", WINDOW_AUTOSIZE);// Create a window for display.
	cv::namedWindow("3.jpg Image", WINDOW_AUTOSIZE);// Create a window for display.
	cv::namedWindow("4.jpg Image", WINDOW_AUTOSIZE);// Create a window for display.
	cv::imshow("1.jpg Image", captionedImage[0]);                   // Show our image inside it.
	cv::imshow("2.jpg Image", captionedImage[1]);                   // Show our image inside it.
	cv::imshow("3.jpg Image", captionedImage[2]);                   // Show our image inside it.
	cv::imshow("4.jpg Image", captionedImage[3]);                   // Show our image inside it.

	cv::waitKey(0);                                          // Wait for a keystroke in the window
}

Mat convertRGB2Greyscale(Mat image)
{
	Mat dst = Mat::zeros(image.size(), CV_8UC1);

	int height = image.size().height;
	int width = image.size().width;

	for (int y = 0; y < height; y++)
		for (int x = 0; x < width; x++)
			dst.at<uchar>(y, x) = RGBPIXEL2GREYSCALE(image.at<Vec3b>(y, x).val[0], image.at<Vec3b>(y, x).val[1], image.at<Vec3b>(y, x).val[2]);
	return dst;
}


Mat convertBinaryImage(Mat image, int &area)
{
	Mat dst = Mat::zeros(image.size(), CV_8UC1);

	int height = image.size().height;
	int width = image.size().width;

	for (int y = 0; y < height; y++)
		for (int x = 0; x < width; x++)
		{
			dst.at<uchar>(y, x) = BINARIZATION(image.at<uchar>(y, x));
			if (BINARIZATION(image.at<uchar>(y, x)) != 0)
			{
				area++; // shape가 포함된 영역이므로 area를 증가
			}
		}

	return dst;
}


/*
Algorithm : 컨투어 추출을 위함 150~210

## square tracing algorithm ##
Input: A square tessellation, T, containing a connected component P of black cells.
Output: A sequence B (b1, b2 ,..., bk) of boundary pixels i.e. the contour.

Begin

Set B to be empty.
From bottom to top and left to right scan the cells of T until a black pixel, s, of P is found.
Insert s in B.
Set the current pixel, p, to be the starting pixel, s.
Turn left i.e. visit the left adjacent pixel of p.
Update p i.e. set it to be the current pixel.

While p not equal to s do
   If the current pixel p is black
		insert p in B and turn left (visit the left adjacent pixel of p).
		Update p i.e. set it to be the current pixel.

   else
		turn right (visit the right adjacent pixel of p).
		Update p i.e. set it to be the current pixel.
end While

End

*/

void setContourStartPoint(Point& p, Mat image)
{
	int height = image.size().height;
	int width = image.size().width;

	for (int row = 0; row < height; row++)
		for (int col = 0; col < width; col++)
		{
			if (image.at<uchar>(row, col) == 255)
			{
				p.x = col;
				p.y = row;
				return;
			}
		}
}

Point GoLeft(Point p) { return Point(p.y, -p.x); }
Point GoRight(Point p) { return Point(-p.y, p.x); }
vector<Point> searchContour(Mat image)
{
	vector<Point> contourPoints;
	// contour image (http://www.imageprocessingplace.com/downloads_V3/root_downloads/tutorials/contour_tracing_Abeer_George_Ghuneim/square.html)
	int height = image.size().height;
	int width = image.size().width;

	Point startPoint(-1, -1);
	setContourStartPoint(startPoint, image);

	if (startPoint.x == -1 && startPoint.y == -1)
	{
		//fail to detect any points
		return contourPoints;
	}
	contourPoints.push_back(startPoint);
	Point nextStep = GoLeft(Point(1, 0));
	Point next = startPoint + nextStep;
	while (next != startPoint) {
		if (image.at<uchar>(next.y, next.x) == 0) {
			nextStep = GoRight(nextStep);
			next = next + nextStep;
		}
		else {
			contourPoints.push_back(next);
			nextStep = GoLeft(nextStep);
			next = next + nextStep;
		}
	}


	if (contourPoints.size() < 5) // Perhaps 8-connected but not 4-connected cause problem http://www.imageprocessingplace.com/downloads_V3/root_downloads/tutorials/contour_tracing_Abeer_George_Ghuneim/8con.html
	{

		for (Point p : contourPoints) {
			image.at<uchar>(p.y, p.x) = 0;
		}
		contourPoints = searchContour(image);
	}
	return contourPoints;

}

double findPerpendicularDistance(Point p,Point p1, Point p2){
	double result;
	double slope;
	double intercept;
	if (p1.x == p2.x) {
		result = fabs(p.x - p1.x);
	}
	else {
		slope = (double)(p2.y - p1.y) / (double)(p2.x - p1.x);
		intercept = (double)p1.y - (slope * p1.x);
		result = fabs(slope * p.x - (double)p.y + intercept) / sqrt(pow(slope, 2) + 1.0);
	}
	return result;
}

vector<Point> findVertex (vector<Point> v)
{
	Point firstPoint = v[0];
	Point lastPoint = v[v.size() - 1];

	if (v.size() < 3) {
		return v;
	}
	int index = -1;
	double maxDist = 0;

	for (int i = 1; i < v.size() - 1; i++) {
		double cDist = findPerpendicularDistance(v[i], firstPoint, lastPoint);
		if (cDist > maxDist) {
			index = i;
			maxDist = cDist;
		}
	}

	vector<Point> ret;

	ret.push_back(firstPoint);
	ret.push_back(v[index]);

	return ret;
}

vector<Point> linePointDist(vector<Point>line, vector<Point> v)
{
	double a, b, c, maxD, currentD;
	maxD = 0;
	currentD = 0;
	if (line[0].x == line[1].x)
	{
		// ax+by+c=0
		// 1*x=k
		a = 1;
		b = 0;
		c = line[0].x;
	}
	else
	{
		//y-mx+line[0].y+mline[0].x=0
		double m = ((double)line[0].y - line[1].y) / ((double)line[0].x-line[1].x);
		a = -m;
		b = 1;
		c = line[0].x * m - line[0].y;
	}
	
	double denominator =sqrt( a*a+b*b);
	int idx = 0;
	int maxIdx = -1;
	for (Point cur : v)
	{
		currentD=fabs(a* cur.x + b * cur.y + c)/denominator;
		if (currentD > maxD)
		{
			maxD = currentD;
			maxIdx = idx;
		}
		idx++;
	}
	
	line.push_back(v[maxIdx]);
	return line;
}

double triangleArea(vector<Point>v)
{
	double firstTerm = (double)v[0].x * v[1].y + v[1].x * v[2].y + v[2].x * v[0].y;
	double secondTerm = (double)v[0].x * v[2].y + v[2].x * v[1].y + v[1].x * v[0].y;

	double ret= (1.0 / 2.0) * fabs(firstTerm - secondTerm);

	return ret;
}



vector<Point> rdp(vector <Point>v, int epsilon) {

	Point firstPoint = v[0];
	Point lastPoint = v[v.size() - 1];

	if (v.size() < 3) {
		return v;
	}
	int index = -1;
	double maxDist = 0;

	for (int i = 1; i < v.size() - 1; i++) {
		double cDist = findPerpendicularDistance(v[i], firstPoint, lastPoint);
		if (cDist > maxDist) {
			index = i;
			maxDist = cDist;
		}
	}
	if (maxDist > epsilon) {
		vector<Point> l1 = vector<Point>(v.begin(), v.begin() + index);
		vector<Point> l2 = vector<Point>(v.begin() + index, v.end());
		vector<Point> r1 = rdp(l1, epsilon);
		vector<Point> r2 = rdp(l2, epsilon);
		vector<Point> rs = r1;
		rs.insert(rs.end(), r2.begin(), r2.end());
		return rs;
	}
	else {
		vector<Point>a{ firstPoint, lastPoint };
		return a;
	}
	return v;
}

void captionTriangle(Mat image, Mat& captionedImage)
{
	int height = image.size().height;
	int width = image.size().width;

	for (int y = 0; y < 10; y++)
		for (int x = 0; x < 101; x++)
		{
			if (captionTriangleArray[y][x])
			{
				image.at<Vec3b>(y, x).val[0] = 255;
				image.at<Vec3b>(y, x).val[1] = 125;
				image.at<Vec3b>(y, x).val[2] = 0;
			}
		
		}

	captionedImage = image;

}
void captionCircle(Mat image, Mat& captionedImage)
{
	int height = image.size().height;
	int width = image.size().width;

	for (int y = 0; y < 10; y++)
		for (int x = 0; x < 75; x++)
		{
			if (captionCircleArray[y][x])
			{
				image.at<Vec3b>(y, x).val[0] = 0;
				image.at<Vec3b>(y, x).val[1] = 125;
				image.at<Vec3b>(y, x).val[2] = 255;
			}

		}

	captionedImage = image;
}