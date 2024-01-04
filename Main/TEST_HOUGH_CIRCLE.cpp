#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

const std::string IMG_PATH = "imgs/d.png";

using namespace cv;
using namespace std;

int main() {
    // 이미지 불러오기
    Mat image = imread(IMG_PATH);

    // 이미지 그레이스케일로 변환
    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);

    // 가우시안 블러 적용
    GaussianBlur(gray, gray, Size(5, 5), 0);

    // 허프 원 변환을 사용하여 원 감지
    vector<Vec3f> circles;
    HoughCircles(gray, circles, HOUGH_GRADIENT, 1, 10, 100, 30, 10, 50);

    // 원을 찾았을 경우
    if (!circles.empty()) {
        for (size_t i = 0; i < circles.size(); i++) {
            Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
            int radius = cvRound(circles[i][2]);

            // 감지된 원의 중심에 원을 그림
            circle(image, center, 3, Scalar(0, 255, 0), -1, 8, 0);

            // 감지된 원의 외곽을 그림
            circle(image, center, radius, Scalar(0, 0, 255), 2, 8, 0);
        }

        // 원을 감지한 후 사각형 그리기 등 추가적인 로직을 수행할 수 있음
        // ...

        // 결과 이미지 표시
        imshow("Detected Circles", image);
        waitKey(0);
    }
    else {
        cout << "No circles detected" << endl;
    }

    return 0;
}