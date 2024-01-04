#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

int main() {
    cv::Mat img = cv::imread("imgs/inb.png");

    cv::Ptr<cv::KAZE> kaze = cv::KAZE::create();
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    kaze->detectAndCompute(img, cv::noArray(), keypoints, descriptors);

    cv::Mat img_keypoints;
    cv::drawKeypoints(img, keypoints, img_keypoints);

    cv::imshow("KAZE Keypoints", img_keypoints);
    cv::waitKey();
    return 0;
}
