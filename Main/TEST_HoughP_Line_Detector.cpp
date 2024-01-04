#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    // 이미지 로드
    string PATH = "imgs/te.png";
    Mat src = imread(PATH);
    double rsizeNum = 0.7;
    resize(src, src, {}, rsizeNum, rsizeNum);

    // 그레이스케일로 변환
    Mat gray;
    cvtColor(src, gray, COLOR_BGR2GRAY);

    // 엣지 검출을 위한 캐니 엣지 적용
    Mat edges;
    Canny(gray, edges, 50, 150, 3);

    // 프로버빌리티 허프 변환을 이용한 직선 검출
    vector<Vec4i> lines;
    HoughLinesP(edges, lines, 1, CV_PI / 180, 40, 500, 30);

    // 검출된 직선 그리기
    for (size_t i = 0; i < lines.size(); i++) {
        Vec4i l = lines[i];
        line(src, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, LINE_AA);
    }

    // 결과 이미지 표시
    imshow("Detected Lines - Probabilistic Hough", src);
    waitKey();

    return 0;
}
