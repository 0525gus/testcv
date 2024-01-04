#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    // �̹��� �ε�
    string PATH = "imgs/te.png";
    Mat src = imread(PATH);
    double rsizeNum = 0.7;
    resize(src, src, {}, rsizeNum, rsizeNum);

    // �׷��̽����Ϸ� ��ȯ
    Mat gray;
    cvtColor(src, gray, COLOR_BGR2GRAY);

    // ���� ������ ���� ĳ�� ���� ����
    Mat edges;
    Canny(gray, edges, 50, 150, 3);

    // ���ι�����Ƽ ���� ��ȯ�� �̿��� ���� ����
    vector<Vec4i> lines;
    HoughLinesP(edges, lines, 1, CV_PI / 180, 40, 500, 30);

    // ����� ���� �׸���
    for (size_t i = 0; i < lines.size(); i++) {
        Vec4i l = lines[i];
        line(src, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, LINE_AA);
    }

    // ��� �̹��� ǥ��
    imshow("Detected Lines - Probabilistic Hough", src);
    waitKey();

    return 0;
}
