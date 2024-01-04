#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

Mat inbox;

int main() {
	string PATH = "imgs/te.png";
	inbox = imread(PATH, 0);  //load as grayScale 
	cv::GaussianBlur(inbox, inbox, cv::Size(7, 7), 0);
	//cv::threshold(inbox, inbox, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);


	Ptr<SIFT> detector = SIFT::create();
	vector<KeyPoint> keypoints;
	detector->detect(inbox, keypoints);

	Mat output;
	drawKeypoints(inbox, keypoints, output);
	namedWindow("inbox");imshow("inbox", output);waitKey(0);

}