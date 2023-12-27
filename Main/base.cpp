#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

// ���� �� ���� �Լ�
void houghLinesCallback(int, void*);

// ���� ����
int rho_slider = 10;  // �ʱ� rho ��
int theta_slider = 235;  // �ʱ� theta ��
int threshold_slider = 100;  // �ʱ� threshold ��

Mat image; // �̹��� ���� ����
Mat edges; // ���� ���� ����

int main() {
    // �̹����� �ε��մϴ�.
    image = imread("dot45.png");

    // �׷��̽����Ϸ� ��ȯ�մϴ�.
    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);

    // ���� ������ �����մϴ�. (��: Canny ���� ����)
    Canny(gray, edges, 50, 150);

    // �ʱ� ���� �� ������ �����մϴ�.
    std::vector<Vec2f> lines;
    HoughLines(edges, lines, rho_slider, CV_PI / (theta_slider ), threshold_slider);

    // ����� ǥ���մϴ�.
    namedWindow("Hough Lines");
    imshow("Hough Lines", image);

    // Ʈ���ٸ� �����ϰ� �ݹ� �Լ��� ����մϴ�.
    createTrackbar("Rho * 10", "Hough Lines", &rho_slider, 100, houghLinesCallback);
    createTrackbar("Theta", "Hough Lines", &theta_slider, 360, houghLinesCallback);
    createTrackbar("Threshold", "Hough Lines", &threshold_slider, 500, houghLinesCallback);

    waitKey(0);

    return 0;
}




void houghLinesCallback(int, void*) {
    // Ʈ������ ���� �Ǽ��� ��ȯ�մϴ�.
    double rho = static_cast<double>(rho_slider) / 10.0;

    // Ʈ������ ���� ����� ������ ���ο� ���� �� ������ �����մϴ�.
    std::vector<Vec2f> lines;
    HoughLines(edges, lines, rho, CV_PI / (theta_slider ), threshold_slider);

    // ����� ���� ���� �̹����� �׸��ϴ�.
    Mat result = image.clone();  // �̹��� ũ��� ä���� ����� �̹����� ����
    for (size_t i = 0; i < lines.size(); ++i) {
        float rho = lines[i][0], theta = lines[i][1];
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a * rho, y0 = b * rho;
        pt1.x = cvRound(x0 + 1000 * (-b));
        pt1.y = cvRound(y0 + 1000 * (a));
        pt2.x = cvRound(x0 - 1000 * (-b));
        pt2.y = cvRound(y0 - 1000 * (a));
        line(result, pt1, pt2, Scalar(0, 0, 255), 2, LINE_AA);
    }

    // ���ο� ����� ǥ���մϴ�.
    imshow("Hough Lines", result);
}
