#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

using namespace cv;
//SBD
int main()
{
    cv::Mat img = cv::imread("te.png", cv::IMREAD_GRAYSCALE); // �̹����� �н��ϴ�.
    GaussianBlur(img, img, Size(5, 5), 0);
    cv::threshold(img, img, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    Canny(img, img, 50, 150);


    // SimpleBlobDetector�� �Ķ���͸� �����մϴ�.
    cv::SimpleBlobDetector::Params params;
    params.filterByArea = true;
    params.minArea = 20; // �ȼ� ���� ���� ����

    // SimpleBlobDetector�� �����մϴ�.
    cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(params);

    // Ư¡���� ã���ϴ�.
    std::vector<cv::KeyPoint> keypoints;
    detector->detect(img, keypoints);

    // ����� ����մϴ�.
    cv::Mat img_with_keypoints;
    cv::drawKeypoints(img, keypoints, img_with_keypoints);
    cv::imshow("Keypoints", img_with_keypoints);
    cv::waitKey(0);

    return 0;
}