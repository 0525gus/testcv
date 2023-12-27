#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

struct Line {
    Point pt1, pt2;
};

void houghLinesCallback(int, void*);
void on_mouse(int event, int x, int y, int flags, void* userdata);
// 트랙바
int rho_slider = 2;
int theta_slider = 180;
int threshold_slider = 108;

Mat img;
Mat edges;

vector<Line> linePoints;
vector<vector<Line>> clusters;

int main() {
    img = imread("a.png");
    double rsizeNum = 1;
    resize(img, img, {}, rsizeNum, rsizeNum);

    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, gray, cv::Size(7, 7), 0);
    cv::threshold(gray, gray, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    Canny(gray, edges, 50, 150);

    namedWindow("Hough Lines");
    imshow("Hough Lines", img);

    // 트랙바 콜백함수 생성
    createTrackbar("Rho * 10", "Hough Lines", &rho_slider, 100, houghLinesCallback);
    createTrackbar("Theta", "Hough Lines", &theta_slider, 360, houghLinesCallback);
    createTrackbar("Threshold", "Hough Lines", &threshold_slider, 500, houghLinesCallback);
    waitKey(0);

    return 0;
}

void houghLinesCallback(int, void*) {
    double rho = static_cast<double>(rho_slider) / 10.0;
    vector<Vec2f> lines;
    HoughLines(edges, lines, rho, CV_PI / (theta_slider), threshold_slider);

    Mat result = img.clone();

    linePoints.clear();
    clusters.clear();

    for (size_t i = 0; i < lines.size(); ++i) {
        float rho = lines[i][0], theta = lines[i][1];
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a * rho, y0 = b * rho;
        pt1.x = cvRound(x0 + 1000 * (-b));
        pt1.y = cvRound(y0 + 1000 * (a));
        pt2.x = cvRound(x0 - 1000 * (-b));
        pt2.y = cvRound(y0 - 1000 * (a));

        linePoints.push_back({ pt1, pt2 });
    }

    // linepoints 기준으로 cluster
    for (const auto& line : linePoints) {
        bool foundCluster = false;
        for (auto& cluster : clusters) {
            const auto& representative = cluster.front();
            if (abs(representative.pt1.x - line.pt1.x) < 10 && abs(representative.pt1.y - line.pt1.y) < 10 &&
                abs(representative.pt2.x - line.pt2.x) < 10 && abs(representative.pt2.y - line.pt2.y) < 10) {
                cluster.push_back(line);
                foundCluster = true;
                break;
            }
        }

        if (!foundCluster) {
            clusters.push_back({ line });
        }
    }

    //cluster line draw
    for (const auto& cluster : clusters) {
        Point pt1(0, 0), pt2(0, 0);
        for (const auto& line : cluster) {
            pt1 += line.pt1;
            pt2 += line.pt2;
        }
        cout << "pt1 = " << pt1 << endl;
        cout << "pt2 =" << pt2 << endl;
        pt1.x /= cluster.size();
        pt1.y /= cluster.size();
        pt2.x /= cluster.size();
        pt2.y /= cluster.size();

        line(result, pt1, pt2, Scalar(0, 255, 0), 2, LINE_AA);
        cout << "Number of lines in this cluster: " << cluster.size() << endl;
    }
    cv::setMouseCallback("Hough Lines", on_mouse);
    imshow("Hough Lines", result);
}

void on_mouse(int event, int x, int y, int flags, void* userdata) {

    if (event == EVENT_LBUTTONDOWN) {
        cout << x << "," << y << endl;
    }
    //구현하는 부분
}