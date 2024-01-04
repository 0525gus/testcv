#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

int main() {
    // �̹��� �ε�
    cv::Mat img = cv::imread("path_to_image.jpg");
    if (img.empty()) {
        std::cerr << "Image load failed!" << std::endl;
        return -1;
    }

    // SURF ����
    int minHessian = 400;
    cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create(minHessian);

    // Ư¡�� ���� �� ����� ���
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    detector->detectAndCompute(img, cv::noArray(), keypoints, descriptors);

    // ��� �̹���
    cv::Mat img_keypoints;
    cv::drawKeypoints(img, keypoints, img_keypoints, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    // �̹��� ǥ��
    cv::imshow("SURF Keypoints", img_keypoints);
    cv::waitKey();

    return 0;
}
s