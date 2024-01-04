#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <vector>

int main() {
    // �̹��� �ε�
    cv::Mat img = cv::imread("imgs/inb.png");
    if (img.empty()) {
        std::cerr << "Image load failed!" << std::endl;
        return -1;
    }

    // SIFT ����
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();

    // Ư¡�� ���� �� ����� ���
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    sift->detectAndCompute(img, cv::noArray(), keypoints, descriptors);

    // ��� �̹���
    cv::Mat img_keypoints;
    cv::drawKeypoints(img, keypoints, img_keypoints, cv::Scalar(0,255,0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    // �̹��� ǥ��
    cv::imshow("SIFT Keypoints", img_keypoints);
    cv::waitKey();

    return 0;
}
