#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <vector>

int main() {
    // 이미지 로드
    cv::Mat img = cv::imread("imgs/inb.png");
    if (img.empty()) {
        std::cerr << "Image load failed!" << std::endl;
        return -1;
    }

    // SIFT 생성
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();

    // 특징점 검출 및 기술자 계산
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    sift->detectAndCompute(img, cv::noArray(), keypoints, descriptors);

    // 결과 이미지
    cv::Mat img_keypoints;
    cv::drawKeypoints(img, keypoints, img_keypoints, cv::Scalar(0,255,0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    // 이미지 표시
    cv::imshow("SIFT Keypoints", img_keypoints);
    cv::waitKey();

    return 0;
}
