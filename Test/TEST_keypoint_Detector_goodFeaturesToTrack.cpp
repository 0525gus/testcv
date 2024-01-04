#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main() {
    string PATH = "imgs/te.png";
    Mat image = imread(PATH, 0);  // 그레이스케일로 로드
    GaussianBlur(image, image, Size(9, 9), 0);
    threshold(image, image, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    Canny(image, image, 0, 0);
    //imshow("Good Features To Track", image);waitKey(0);

    if (image.empty()) {
        cerr << "이미지없음." << endl;
        return -1;
    }

    // 코너 찾기
    vector<Point2f> corners;
    //goodFeaturesToTrack(image, corners, 200, 0.01, 10, {}, true, {}, 0.04);
    goodFeaturesToTrack(image, corners, 200, 0.01, 10);

    // 결과 표시
    Mat output = image.clone();
    cvtColor(output, output, COLOR_GRAY2BGR);
    for (const auto& corner : corners) {
        circle(output, corner, 5, Scalar(0, 0, 255), -1);
    }

    namedWindow("Good Features To Track");
    imshow("Good Features To Track", output);
    waitKey(0);

    return 0;
}