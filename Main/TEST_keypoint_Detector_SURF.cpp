#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

int main() {
    // 이미지 로드
    cv::Mat img = cv::imread("path_to_image.jpg");
    if (img.empty()) {
        std::cerr << "Image load failed!" << std::endl;
        return -1;
    }

    // SURF 생성
    int minHessian = 400;
    cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create(minHessian);

    // 특징점 검출 및 기술자 계산
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    detector->detectAndCompute(img, cv::noArray(), keypoints, descriptors);

    // 결과 이미지
    cv::Mat img_keypoints;
    cv::drawKeypoints(img, keypoints, img_keypoints, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    // 이미지 표시
    cv::imshow("SURF Keypoints", img_keypoints);
    cv::waitKey();

    return 0;
}
s