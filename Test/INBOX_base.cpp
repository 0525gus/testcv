#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

int main() {
    // 이미지 불러오기
    Mat original = imread("imgs/dd.PNG");
    Mat target = imread("imgs/dT.jpg", IMREAD_GRAYSCALE);

    // 이미지 크기 조정
    resize(target, target, {}, 0.3, 0.3);
    resize(original, original, {}, 0.7, 0.7);

    // 이미지 불러오기 확인
    if (original.empty() || target.empty()) {
        cout << "이미지를 불러오는 데 실패했습니다." << endl;
        return -1;
    }

    // 전처리 - 가우시안 블러와 쓰레시홀딩
    GaussianBlur(target, target, Size(9, 9), 0);
    threshold(target, target, 0, 255, THRESH_BINARY | THRESH_OTSU);
    GaussianBlur(original, original, Size(9, 9), 0);


    cv::imshow("2", target);
    waitKey(0);

    // Canny 엣지 검출

    // KAZE 특징 검출기 생성
    Ptr<KAZE> detector = KAZE::create();

    // 키 포인트와 디스크립터 추출
    vector<KeyPoint> keypointsOriginal, keypointsTarget;
    Mat descriptorsOriginal, descriptorsTarget;
    detector->detectAndCompute(original, noArray(), keypointsOriginal, descriptorsOriginal);
    detector->detectAndCompute(target, noArray(), keypointsTarget, descriptorsTarget);

    // BFMatcher로 매칭 수행
    BFMatcher matcher(NORM_L2);
    vector<vector<DMatch>> knnMatches;
    matcher.knnMatch(descriptorsOriginal, descriptorsTarget, knnMatches, 2);

    // Lowe의 비율 테스트를 사용하여 좋은 매칭 필터링
    const float ratio_thresh = 0.80f;
    vector<DMatch> goodMatches;
    for (size_t i = 0; i < knnMatches.size(); i++) {
        if (knnMatches[i][0].distance < ratio_thresh * knnMatches[i][1].distance) {
            goodMatches.push_back(knnMatches[i][0]);
        }
    }

    // 변환 행렬 계산
    vector<Point2f> obj, scene;
    for (size_t i = 0; i < goodMatches.size(); i++) {
        obj.push_back(keypointsOriginal[goodMatches[i].queryIdx].pt);
        scene.push_back(keypointsTarget[goodMatches[i].trainIdx].pt);
    }
    Mat H = findHomography(obj, scene, RANSAC);

    // 사각형의 윤곽 찾기
    vector<Point2f> objCorners(4);
    objCorners[0] = Point2f(0, 0);
    objCorners[1] = Point2f((float)original.cols, 0);
    objCorners[2] = Point2f((float)original.cols, (float)original.rows);
    objCorners[3] = Point2f(0, (float)original.rows);
    vector<Point2f> sceneCorners(4);
    perspectiveTransform(objCorners, sceneCorners, H);

    // 윤곽 그리기
    line(target, sceneCorners[0], sceneCorners[1], Scalar(0, 255, 0), 4);
    line(target, sceneCorners[1], sceneCorners[2], Scalar(0, 255, 0), 4);
    line(target, sceneCorners[2], sceneCorners[3], Scalar(0, 255, 0), 4);
    line(target, sceneCorners[3], sceneCorners[0], Scalar(0, 255, 0), 4);

    // 결과 표시
    imshow("Original Image", original);
    imshow("Target Image with Detected Contour", target);
    waitKey(0);

    return 0;
}
