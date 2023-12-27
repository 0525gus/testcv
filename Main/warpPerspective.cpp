#include <iostream>
#include "opencv2/opencv.hpp"

cv::Mat src;


cv::Point2f srcQuad[4], dstQuad[4]; //src 행렬과 dst 행렬 선언부분

void on_mouse(int event, int x, int y, int flags, void* userdata)
{
    static int cnt = 0;

    if (event == cv::EVENT_LBUTTONDOWN) {
        if (cnt < 4) {
            srcQuad[cnt++] = cv::Point2f(x, y);

            cv::circle(src, cv::Point(x, y), 5, cv::Scalar(255, 255, 0), -1);
            cv::imshow("src", src);

            if (cnt == 4) {
                int w = 400, h = 400; // 정사각형으로 출력되게 만듬


                //역디귿 모양으로 찍어야 보이는대로 출력됨
                //코너를 정함
                dstQuad[0] = cv::Point2f(0, 0);
                dstQuad[1] = cv::Point2f(w - 1, 0);
                dstQuad[2] = cv::Point2f(w - 1, h - 1);
                dstQuad[3] = cv::Point2f(0, h - 1);

                cv::Mat pers = cv::getPerspectiveTransform(srcQuad, dstQuad);//원근변환

                cv::Mat dst;
                cv::warpPerspective(src, dst, pers, cv::Size(w, h));

                cv::imshow("dst", dst);
            }
        }
    }
}

int main()
{
    src = cv::imread("dot_book1.jpg", cv::IMREAD_GRAYSCALE);
    if (src.empty()) { //못 읽었을시 예외처리
        std::cout << "Image load failed!\n";
        return -1;
    }

    // 가우시안 블러 
    cv::GaussianBlur(src, src, cv::Size(5, 5), 0);

    // 쓰레스홀드 + 이진화
    //int threshold_value = 150; //임계값 산정 -> 세번째 파라미터에 값 넣기 (수동)
     cv::threshold(src, src,  0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU); 
    //THRESH_OTSU -> 자동임계값 선택 알고리즘 


    // canny edge 
    cv::Canny(src, src, 50, 100);
     std::vector<cv::Vec2f> lines;
     //cv::HoughLinesP(src, lines, 1, CV_PI / 180, 100);  // HoughLinesP 함수를 사용할 수도 있음

     //for (size_t i = 0; i < lines.size(); ++i) {
     //    float rho = lines[i][0];
     //    float theta = lines[i][1];
     //    if (theta > CV_PI / 4 && theta < 3 * CV_PI / 4) {
     //        cv::Point pt1, pt2;
     //        double a = cos(theta), b = sin(theta);
     //        double x0 = a * rho, y0 = b * rho;
     //        pt1.x = cvRound(x0 + 1000 * (-b));
     //        pt1.y = cvRound(y0 + 1000 * (a));
     //        pt2.x = cvRound(x0 - 1000 * (-b));
     //        pt2.y = cvRound(y0 - 1000 * (a));
     //        line(src, pt1, pt2, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
     //    }
     //}
    cv::namedWindow("src", cv::WINDOW_AUTOSIZE);

    cv::setMouseCallback("src", on_mouse);//이곳

    double scale = 0.3;
    cv::resize(src, src, {}, scale, scale);

    cv::imshow("src", src);
    cv::waitKey(0);

    cv::destroyAllWindows();
}