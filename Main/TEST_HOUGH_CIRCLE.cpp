#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

const std::string IMG_PATH = "imgs/d.png";

using namespace cv;
using namespace std;

int main() {
    // �̹��� �ҷ�����
    Mat image = imread(IMG_PATH);

    // �̹��� �׷��̽����Ϸ� ��ȯ
    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);

    // ����þ� �� ����
    GaussianBlur(gray, gray, Size(5, 5), 0);

    // ���� �� ��ȯ�� ����Ͽ� �� ����
    vector<Vec3f> circles;
    HoughCircles(gray, circles, HOUGH_GRADIENT, 1, 10, 100, 30, 10, 50);

    // ���� ã���� ���
    if (!circles.empty()) {
        for (size_t i = 0; i < circles.size(); i++) {
            Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
            int radius = cvRound(circles[i][2]);

            // ������ ���� �߽ɿ� ���� �׸�
            circle(image, center, 3, Scalar(0, 255, 0), -1, 8, 0);

            // ������ ���� �ܰ��� �׸�
            circle(image, center, radius, Scalar(0, 0, 255), 2, 8, 0);
        }

        // ���� ������ �� �簢�� �׸��� �� �߰����� ������ ������ �� ����
        // ...

        // ��� �̹��� ǥ��
        imshow("Detected Circles", image);
        waitKey(0);
    }
    else {
        cout << "No circles detected" << endl;
    }

    return 0;
}