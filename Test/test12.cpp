#include <opencv2/opencv.hpp>
#include <vector>
#include <chrono>
#include <format>
//#include <opencv2/features2d.hpp>

using namespace cv;
using namespace std;
cv::Point2f srcQuad[4], dstQuad[4]; //src 행렬과 dst 행렬 선언부분

bool isRectangleValid(const vector<Point2f>& corners) {
    if (corners.size() != 4) return false;

    // 사각형을 이루는 네 점 사이의 각도 확인
    for (int i = 0; i < 4; i++) {
        Point2f p1 = corners[i];
        Point2f p2 = corners[(i + 1) % 4];
        Point2f p3 = corners[(i + 2) % 4];

        // 벡터 계산
        Point2f v1 = p2 - p1;
        Point2f v2 = p3 - p2;

        // 각도 계산
        double angle = atan2(v1.y, v1.x) - atan2(v2.y, v2.x);

        // 라디안을 도로 변환
        angle = abs(angle * 180.0 / CV_PI);

        // 각도가 90도에 가까운지 확인
        if (angle < 80 || angle > 100) return false;
    }
    return true;

}
int main() {
    // 이미지 불러오기
    auto start = std::chrono::high_resolution_clock::now();

    Mat original = imread("imgs/dd.PNG");
    Mat target = imread("imgs/dT.jpg", IMREAD_GRAYSCALE);

    // 타겟 이미지 크기 조정 및 전처리
    resize(target, target, {}, 0.2, 0.2);
    GaussianBlur(target, target, Size(7, 7), 0);
    //threshold(target, target, 0, 255, THRESH_BINARY | THRESH_OTSU);
    threshold(target, target, 100, 200, THRESH_BINARY);

    Mat* target_p = &target;
    cout << target_p << endl;


    //cv::imshow("test", target);cv::waitKey(0);

    // KAZE 특징 검출기 생성
    //Ptr<KAZE> detector = KAZE::create();
    Ptr<SIFT> detector = SIFT::create();
    //Ptr<AKAZE> detector = AKAZE::create();




    // 원본 이미지 스케일 조정
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
        // 원본 이미지 크기 조정
        resize(original, scaledOriginal, Size(), scale, scale);

        // 키 포인트와 디스크립터 추출
        detector->detectAndCompute(scaledOriginal, noArray(), keypointsOriginal, descriptorsOriginal);
        detector->detectAndCompute(target, noArray(), keypointsTarget, descriptorsTarget);

        // BFMatcher로 매칭 수행
        matcher.knnMatch(descriptorsOriginal, descriptorsTarget, knnMatches, 2);

        // Lowe의 비율 테스트를 사용하여 좋은 매칭 필터링
        const float ratio_thresh = 0.80f;
        goodMatches.clear();
        for (size_t i = 0; i < knnMatches.size(); i++) {
            if (knnMatches[i][0].distance < ratio_thresh * knnMatches[i][1].distance) {
                goodMatches.push_back(knnMatches[i][0]);
            }
        }

        // 충분한 좋은 매칭이 있는지 검사
        if (goodMatches.size() >= 4) {
            // 변환 행렬 계산 시도
            try {
                obj.clear();
                scene.clear();
                for (size_t i = 0; i < goodMatches.size(); i++) {
                    obj.push_back(keypointsOriginal[goodMatches[i].queryIdx].pt);
                    scene.push_back(keypointsTarget[goodMatches[i].trainIdx].pt);
                }
                Mat H = findHomography(obj, scene, RANSAC);

                // 사각형의 윤곽 찾기
                vector<Point2f> objCorners(4);
                objCorners[0] = Point2f(0, 0);
                objCorners[1] = Point2f((float)scaledOriginal.cols, 0);
                objCorners[2] = Point2f((float)scaledOriginal.cols, (float)scaledOriginal.rows);
                objCorners[3] = Point2f(0, (float)scaledOriginal.rows);
                perspectiveTransform(objCorners, sceneCorners, H);

                if (isRectangleValid(sceneCorners)) {
                    // 윤곽 그리기
                    line(target, sceneCorners[0], sceneCorners[1], Scalar(0, 255, 0), 10);
                    line(target, sceneCorners[1], sceneCorners[2], Scalar(0, 255, 0), 10);
                    line(target, sceneCorners[2], sceneCorners[3], Scalar(0, 255, 0), 10);
                    line(target, sceneCorners[3], sceneCorners[0], Scalar(0, 255, 0), 10);
                    homographyFound = true;
                }
                else {
                    cout << "못찾음" << endl;
                }
                homographyFound = true;

            }
            catch (const cv::Exception& e) {
                cerr << "findHomography에서 예외 발생: " << e.what() << endl;
            }
        }

        // 다음 스케일을 위해 1.5배 증가
        scale *= 1.3;

        // 스케일이 너무 크게 증가하지 않도록 제한
        if (scale > 3.0) break;
    }

    if (!homographyFound) {
        cout << "적절한 매칭을 찾을 수 없습니다." << endl;
        return -1;
    }

    // 결과 표시
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
    cv::Mat pers = cv::getPerspectiveTransform(srcQuad, dstQuad);//원근변환
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
