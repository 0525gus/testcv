#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

int main() {
    // �̹��� �ҷ�����
    Mat original = imread("imgs/dd.PNG");
    Mat target = imread("imgs/dT.jpg", IMREAD_GRAYSCALE);

    // �̹��� ũ�� ����
    resize(target, target, {}, 0.3, 0.3);
    resize(original, original, {}, 0.7, 0.7);

    // �̹��� �ҷ����� Ȯ��
    if (original.empty() || target.empty()) {
        cout << "�̹����� �ҷ����� �� �����߽��ϴ�." << endl;
        return -1;
    }

    // ��ó�� - ����þ� ���� ������Ȧ��
    GaussianBlur(target, target, Size(9, 9), 0);
    threshold(target, target, 0, 255, THRESH_BINARY | THRESH_OTSU);
    GaussianBlur(original, original, Size(9, 9), 0);


    cv::imshow("2", target);
    waitKey(0);

    // Canny ���� ����

    // KAZE Ư¡ ����� ����
    Ptr<KAZE> detector = KAZE::create();

    // Ű ����Ʈ�� ��ũ���� ����
    vector<KeyPoint> keypointsOriginal, keypointsTarget;
    Mat descriptorsOriginal, descriptorsTarget;
    detector->detectAndCompute(original, noArray(), keypointsOriginal, descriptorsOriginal);
    detector->detectAndCompute(target, noArray(), keypointsTarget, descriptorsTarget);

    // BFMatcher�� ��Ī ����
    BFMatcher matcher(NORM_L2);
    vector<vector<DMatch>> knnMatches;
    matcher.knnMatch(descriptorsOriginal, descriptorsTarget, knnMatches, 2);

    // Lowe�� ���� �׽�Ʈ�� ����Ͽ� ���� ��Ī ���͸�
    const float ratio_thresh = 0.80f;
    vector<DMatch> goodMatches;
    for (size_t i = 0; i < knnMatches.size(); i++) {
        if (knnMatches[i][0].distance < ratio_thresh * knnMatches[i][1].distance) {
            goodMatches.push_back(knnMatches[i][0]);
        }
    }

    // ��ȯ ��� ���
    vector<Point2f> obj, scene;
    for (size_t i = 0; i < goodMatches.size(); i++) {
        obj.push_back(keypointsOriginal[goodMatches[i].queryIdx].pt);
        scene.push_back(keypointsTarget[goodMatches[i].trainIdx].pt);
    }
    Mat H = findHomography(obj, scene, RANSAC);

    // �簢���� ���� ã��
    vector<Point2f> objCorners(4);
    objCorners[0] = Point2f(0, 0);
    objCorners[1] = Point2f((float)original.cols, 0);
    objCorners[2] = Point2f((float)original.cols, (float)original.rows);
    objCorners[3] = Point2f(0, (float)original.rows);
    vector<Point2f> sceneCorners(4);
    perspectiveTransform(objCorners, sceneCorners, H);

    // ���� �׸���
    line(target, sceneCorners[0], sceneCorners[1], Scalar(0, 255, 0), 4);
    line(target, sceneCorners[1], sceneCorners[2], Scalar(0, 255, 0), 4);
    line(target, sceneCorners[2], sceneCorners[3], Scalar(0, 255, 0), 4);
    line(target, sceneCorners[3], sceneCorners[0], Scalar(0, 255, 0), 4);

    // ��� ǥ��
    imshow("Original Image", original);
    imshow("Target Image with Detected Contour", target);
    waitKey(0);

    return 0;
}
