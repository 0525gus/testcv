#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

using namespace cv;
//SBD
int main()
{
    cv::Mat img = cv::imread("te.png", cv::IMREAD_GRAYSCALE); // 이미지를 읽습니다.
    GaussianBlur(img, img, Size(5, 5), 0);
    cv::threshold(img, img, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    Canny(img, img, 50, 150);


    // SimpleBlobDetector의 파라미터를 설정합니다.
    cv::SimpleBlobDetector::Params params;
    params.filterByArea = true;
    params.minArea = 20; // 픽셀 수에 따라 조절

    // SimpleBlobDetector를 생성합니다.
    cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(params);

    // 특징점을 찾습니다.
    std::vector<cv::KeyPoint> keypoints;
    detector->detect(img, keypoints);

    // 결과를 출력합니다.
    cv::Mat img_with_keypoints;
    cv::drawKeypoints(img, keypoints, img_with_keypoints);
    cv::imshow("Keypoints", img_with_keypoints);
    cv::waitKey(0);

    return 0;
}