#include <opencv2/opencv.hpp>
#include <vector>
#include <chrono>
#include <format>
//#include <opencv2/features2d.hpp>

using namespace cv;
using namespace std;
cv::Point2f srcQuad[4], dstQuad[4]; //src ��İ� dst ��� ����κ�

bool isRectangleValid(const vector<Point2f>& corners) {
    if (corners.size() != 4) return false;

    // �簢���� �̷�� �� �� ������ ���� Ȯ��
    for (int i = 0; i < 4; i++) {
        Point2f p1 = corners[i];
        Point2f p2 = corners[(i + 1) % 4];
        Point2f p3 = corners[(i + 2) % 4];

        // ���� ���
        Point2f v1 = p2 - p1;
        Point2f v2 = p3 - p2;

        // ���� ���
        double angle = atan2(v1.y, v1.x) - atan2(v2.y, v2.x);

        // ������ ���� ��ȯ
        angle = abs(angle * 180.0 / CV_PI);

        // ������ 90���� ������� Ȯ��
        if (angle < 80 || angle > 100) return false;
    }
    return true;

}
int main() {
    // �̹��� �ҷ�����
    auto start = std::chrono::high_resolution_clock::now();

    Mat original = imread("imgs/dd.PNG");
    Mat target = imread("imgs/dT.jpg", IMREAD_GRAYSCALE);

    // Ÿ�� �̹��� ũ�� ���� �� ��ó��
    resize(target, target, {}, 0.2, 0.2);
    GaussianBlur(target, target, Size(7, 7), 0);
    //threshold(target, target, 0, 255, THRESH_BINARY | THRESH_OTSU);
    threshold(target, target, 100, 200, THRESH_BINARY);

    Mat* target_p = &target;
    cout << target_p << endl;


    //cv::imshow("test", target);cv::waitKey(0);

    // KAZE Ư¡ ����� ����
    //Ptr<KAZE> detector = KAZE::create();
    Ptr<SIFT> detector = SIFT::create();
    //Ptr<AKAZE> detector = AKAZE::create();




    // ���� �̹��� ������ ����
    double scale = 0.7;
    bool homographyFound = false;
    Mat scaledOriginal, descriptorsOriginal, descriptorsTarget;
    vector<KeyPoint> keypointsOriginal, keypointsTarget;
    BFMatcher matcher(NORM_L2);
    vector<vector<DMatch>> knnMatches;
    vector<DMatch> goodMatches;
    vector<Point2f> obj, scene;
    vector<Point2f>sceneCorners(4);
    while (!homographyFound) {
        // ���� �̹��� ũ�� ����
        resize(original, scaledOriginal, Size(), scale, scale);

        // Ű ����Ʈ�� ��ũ���� ����
        detector->detectAndCompute(scaledOriginal, noArray(), keypointsOriginal, descriptorsOriginal);
        detector->detectAndCompute(target, noArray(), keypointsTarget, descriptorsTarget);

        // BFMatcher�� ��Ī ����
        matcher.knnMatch(descriptorsOriginal, descriptorsTarget, knnMatches, 2);

        // Lowe�� ���� �׽�Ʈ�� ����Ͽ� ���� ��Ī ���͸�
        const float ratio_thresh = 0.80f;
        goodMatches.clear();
        for (size_t i = 0; i < knnMatches.size(); i++) {
            if (knnMatches[i][0].distance < ratio_thresh * knnMatches[i][1].distance) {
                goodMatches.push_back(knnMatches[i][0]);
            }
        }

        // ����� ���� ��Ī�� �ִ��� �˻�
        if (goodMatches.size() >= 4) {
            // ��ȯ ��� ��� �õ�
            try {
                obj.clear();
                scene.clear();
                for (size_t i = 0; i < goodMatches.size(); i++) {
                    obj.push_back(keypointsOriginal[goodMatches[i].queryIdx].pt);
                    scene.push_back(keypointsTarget[goodMatches[i].trainIdx].pt);
                }
                Mat H = findHomography(obj, scene, RANSAC);

                // �簢���� ���� ã��
                vector<Point2f> objCorners(4);
                objCorners[0] = Point2f(0, 0);
                objCorners[1] = Point2f((float)scaledOriginal.cols, 0);
                objCorners[2] = Point2f((float)scaledOriginal.cols, (float)scaledOriginal.rows);
                objCorners[3] = Point2f(0, (float)scaledOriginal.rows);
                perspectiveTransform(objCorners, sceneCorners, H);

                if (isRectangleValid(sceneCorners)) {
                    // ���� �׸���
                    line(target, sceneCorners[0], sceneCorners[1], Scalar(0, 255, 0), 10);
                    line(target, sceneCorners[1], sceneCorners[2], Scalar(0, 255, 0), 10);
                    line(target, sceneCorners[2], sceneCorners[3], Scalar(0, 255, 0), 10);
                    line(target, sceneCorners[3], sceneCorners[0], Scalar(0, 255, 0), 10);
                    homographyFound = true;
                }
                else {
                    cout << "��ã��" << endl;
                }
                homographyFound = true;

            }
            catch (const cv::Exception& e) {
                cerr << "findHomography���� ���� �߻�: " << e.what() << endl;
            }
        }

        // ���� �������� ���� 1.5�� ����
        scale *= 1.3;

        // �������� �ʹ� ũ�� �������� �ʵ��� ����
        if (scale > 3.0) break;
    }

    if (!homographyFound) {
        cout << "������ ��Ī�� ã�� �� �����ϴ�." << endl;
        return -1;
    }

    // ��� ǥ��
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "time = " << elapsed.count() << "second\n";
    cv::imshow("Original Image", scaledOriginal);
    imshow("Target Image with Detected Contour", target);
    waitKey(0);



    for (int i = 0; i < 4; i++) {
        srcQuad[i] = cv::Point2f(sceneCorners[i].x, sceneCorners[i].y);
    }
    int w = 400, h = 400;
    dstQuad[0] = cv::Point2f(0, 0);
    dstQuad[1] = cv::Point2f(w - 1, 0);
    dstQuad[2] = cv::Point2f(w - 1, h - 1);
    dstQuad[3] = cv::Point2f(0, h - 1);
    cv::Mat pers = cv::getPerspectiveTransform(srcQuad, dstQuad);//���ٺ�ȯ
    cv::Mat dst;
    cv::warpPerspective(target, dst, pers, cv::Size(w, h));
    cv::imshow("dst", dst);
    waitKey(0);

    std::vector<cv::KeyPoint> keypoints;
    cv::SimpleBlobDetector::Params params;
    params.filterByArea = true;
    params.minArea = 30;

    cv::Ptr<cv::SimpleBlobDetector> detector_Blob = cv::SimpleBlobDetector::create(params);
    detector_Blob->detect(dst, keypoints);

    cv::Mat dst_with_keypoints;
    cv::drawKeypoints(dst, keypoints, dst_with_keypoints, Scalar(0, 255, 0));
    cv::imshow("Keypoints", dst_with_keypoints);waitKey(0);





    return 0;
}
